# trace_agent.py
import uuid
import json
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, AIMessage
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGeneration, LLMResult

# ──────────────────────────────────────────────────────────


class LangChainEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle UUID objects
        if isinstance(obj, uuid.UUID):
            return str(obj)

        # Handle LangChain message objects
        if isinstance(obj, BaseMessage):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
                "additional_kwargs": obj.additional_kwargs
            }

        # Handle other potential LangChain objects
        if hasattr(obj, "to_json"):
            return obj.to_json()

        if hasattr(obj, "dict"):
            return obj.dict()

        # Let the base class handle it or raise TypeError
        return super().default(obj)

# ──────────────────────────────────────────────────────────


class TraceCollector(BaseCallbackHandler):
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.start_ts = datetime.utcnow().isoformat() + "Z"
        self.spans = {}
        self.tree = defaultdict(list)
        self.run_metadata = {}

    def _serialize_message(self, message) -> Dict[str, Any]:
        """Serialize LangChain messages to dict format"""
        if isinstance(message, BaseMessage):
            return {
                "type": message.__class__.__name__.replace("Message", "").lower(),
                "content": message.content,
                "additional_kwargs": getattr(message, 'additional_kwargs', {}),
                "response_metadata": getattr(message, 'response_metadata', {}),
                "name": getattr(message, 'name', None),
                "id": getattr(message, 'id', None),
                "example": getattr(message, 'example', False)
            }
        elif isinstance(message, dict):
            return message
        return {"content": str(message), "type": "unknown"}

    def _serialize_messages(self, messages) -> List[Dict[str, Any]]:
        """Serialize list of messages"""
        if not messages:
            return []

        if isinstance(messages, list):
            return [self._serialize_message(msg) for msg in messages]
        return [self._serialize_message(messages)]

    def _serialize_llm_result(self, result: LLMResult) -> Dict[str, Any]:
        """Serialize LLM result with detailed information"""
        serialized = {
            "generations": []
        }

        if hasattr(result, 'generations') and result.generations:
            for generation_list in result.generations:
                gen_list = []
                for gen in generation_list:
                    if isinstance(gen, ChatGeneration):
                        gen_dict = {
                            "lc": 1,
                            "type": "constructor",
                            "id": ["langchain", "schema", "output", "ChatGeneration"],
                            "kwargs": {
                                "generation_info": getattr(gen, 'generation_info', {}),
                                "type": "ChatGeneration",
                                "message": self._serialize_message(gen.message)
                            }
                        }
                    else:
                        gen_dict = {
                            "text": getattr(gen, 'text', ''),
                            "generation_info": getattr(gen, 'generation_info', {})
                        }
                    gen_list.append(gen_dict)
                serialized["generations"].append(gen_list)

        # Add LLM run info
        if hasattr(result, 'llm_output') and result.llm_output:
            serialized["llm_output"] = result.llm_output

        return serialized

    def _extract_chain_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format chain input"""
        if not inputs:
            return {}

        if not isinstance(inputs, dict):
            if hasattr(inputs, "to_dict"):
                inputs = inputs.to_dict()
            elif hasattr(inputs, "dict"):
                inputs = inputs.dict()
            else:
                return {"input": str(inputs)}

        formatted_input = {}

        # Handle messages specifically
        if "messages" in inputs:
            formatted_input["messages"] = self._serialize_messages(
                inputs["messages"])

        # Handle other inputs
        for key, value in inputs.items():
            if key == "messages":
                continue  # Already handled
            elif isinstance(value, list) and value and hasattr(value[0], '__class__') and 'Message' in value[0].__class__.__name__:
                formatted_input[key] = self._serialize_messages(value)
            else:
                formatted_input[key] = value

        return formatted_input

    def _new_span(self, run_id: str, parent_id: Optional[str], span_type: str = "unknown",
                  input_data: Any = None, name: Optional[str] = None, **kwargs):
        """Create a new span with comprehensive data"""
        span = {
            "id": str(run_id),
            "parent_id": str(parent_id) if parent_id else None,
            "type": span_type,
            "name": name or span_type,
            "start_time": time.time(),
            "input": input_data,
            "output": None,
            "error": None,
            "children": [],
            "duration_ms": None,
            "metadata": kwargs.get('metadata', {})
        }

        self.spans[str(run_id)] = span
        if parent_id:
            self.tree[str(parent_id)].append(str(run_id))

    def _update_span(self, run_id: str, **kwargs):
        """Update span with output and timing information"""
        span = self.spans.get(str(run_id))
        if span:
            span.update(**kwargs)
            span["end_time"] = time.time()
            span["duration_ms"] = round(
                (span["end_time"] - span["start_time"]) * 1000, 2
            )

    # ============ Chain Callbacks ============
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any],
                       run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None,
                       tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Handle chain start with detailed input processing"""

        # Determine chain type and name
        chain_type = "chain"
        chain_name = "chain"

        if isinstance(serialized, dict):
            chain_type = serialized.get(
                "id", ["unknown"])[-1] if serialized.get("id") else "chain"
            chain_name = serialized.get("name", chain_type)

        # Process inputs
        formatted_input = self._extract_chain_input(inputs)

        self._new_span(
            run_id, parent_run_id,
            span_type=chain_type,
            input_data=formatted_input,
            name=chain_name,
            metadata=metadata or {},
            tags=tags or []
        )

    def on_chain_end(self, outputs: Dict[str, Any], run_id: uuid.UUID,
                     parent_run_id: Optional[uuid.UUID] = None, **kwargs):
        """Handle chain end with detailed output processing"""

        formatted_output = {}

        # Handle different output formats
        if isinstance(outputs, dict):
            if "messages" in outputs:
                formatted_output["messages"] = self._serialize_messages(
                    outputs["messages"])

            for key, value in outputs.items():
                if key == "messages":
                    continue  # Already handled
                elif isinstance(value, list) and value and hasattr(value[0], '__class__') and 'Message' in str(value[0].__class__):
                    formatted_output[key] = self._serialize_messages(value)
                else:
                    formatted_output[key] = value
        else:
            formatted_output = outputs

        self._update_span(str(run_id), output=formatted_output)

    def on_chain_error(self, error: Exception, run_id: uuid.UUID,
                       parent_run_id: Optional[uuid.UUID] = None, **kwargs):
        """Handle chain errors"""
        self._update_span(str(run_id), error=str(error))

    # ============ LLM Callbacks ============
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                     run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None,
                     tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Handle LLM start with prompt information"""

        input_data = {"prompts": prompts} if prompts else {}

        self._new_span(
            run_id, parent_run_id,
            span_type="llm",
            input_data=input_data,
            name="llm",
            metadata=metadata or {},
            tags=tags or []
        )

    def on_llm_end(self, response: LLMResult, run_id: uuid.UUID,
                   parent_run_id: Optional[uuid.UUID] = None, **kwargs):
        """Handle LLM end with detailed response information"""

        output_data = self._serialize_llm_result(response)
        self._update_span(str(run_id), output=output_data)

    def on_llm_error(self, error: Exception, run_id: uuid.UUID,
                     parent_run_id: Optional[uuid.UUID] = None, **kwargs):
        """Handle LLM errors"""
        self._update_span(str(run_id), error=str(error))

    # ============ Tool Callbacks ============
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                      run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None,
                      tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Handle tool start"""

        tool_name = "tool"
        if isinstance(serialized, dict):
            tool_name = serialized.get("name", "tool")

        self._new_span(
            run_id, parent_run_id,
            span_type="tool",
            input_data={"input": input_str},
            name=tool_name,
            metadata=metadata or {},
            tags=tags or []
        )

    def on_tool_end(self, output: str, run_id: uuid.UUID,
                    parent_run_id: Optional[uuid.UUID] = None, **kwargs):
        """Handle tool end"""
        self._update_span(str(run_id), output={"output": output})

    def on_tool_error(self, error: Exception, run_id: uuid.UUID,
                      parent_run_id: Optional[uuid.UUID] = None, **kwargs):
        """Handle tool errors"""
        self._update_span(str(run_id), error=str(error))

    # ============ Additional Callbacks ============
    def on_retriever_start(self, serialized: Dict[str, Any], query: str,
                           run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None,
                           tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Handle retriever start"""
        self._new_span(
            run_id, parent_run_id,
            span_type="retriever",
            input_data={"query": query},
            name="retriever",
            metadata=metadata or {},
            tags=tags or []
        )

    def on_retriever_end(self, documents: List[Any], run_id: uuid.UUID,
                         parent_run_id: Optional[uuid.UUID] = None, **kwargs):
        """Handle retriever end"""
        # Serialize documents
        doc_output = []
        for doc in documents:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                doc_output.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "type": "Document"
                })
            else:
                doc_output.append(str(doc))

        self._update_span(str(run_id), output={"documents": doc_output})

    def on_retriever_error(self, error: Exception, run_id: uuid.UUID,
                           parent_run_id: Optional[uuid.UUID] = None, **kwargs):
        """Handle retriever errors"""
        self._update_span(str(run_id), error=str(error))

    # ============ Export Methods ============
    def build_tree_structure(self) -> List[Dict[str, Any]]:
        """Build hierarchical tree structure of spans"""
        def build_span(span_id: str) -> Dict[str, Any]:
            span = self.spans[span_id].copy()

            # Remove timing fields from final output (keep duration_ms)
            span.pop("start_time", None)
            span.pop("end_time", None)

            # Add children
            children = [build_span(child_id)
                        for child_id in self.tree.get(span_id, [])]
            if children:
                span["children"] = children

            return span

        # Find root spans (spans with no parent)
        root_spans = [build_span(span_id) for span_id, span in self.spans.items()
                      if span["parent_id"] is None]

        return root_spans

    def export_trace(self) -> Dict[str, Any]:
        """Export complete trace in Langfuse-compatible format"""
        return {
            "id": self.trace_id,
            "timestamp": self.start_ts,
            "root_spans": self.build_tree_structure()
        }

    def export_detailed_trace(self) -> Dict[str, Any]:
        """Export trace with additional metadata and statistics"""
        trace = self.export_trace()

        # Add statistics
        trace["metadata"] = {
            "total_spans": len(self.spans),
            "span_types": list(set(span["type"] for span in self.spans.values())),
            "total_duration_ms": sum(span.get("duration_ms", 0) for span in self.spans.values()),
            "error_count": len([span for span in self.spans.values() if span.get("error")])
        }

        return trace

    def print_trace(self, detailed: bool = False):
        """Print trace in JSON format"""
        trace_data = self.export_detailed_trace() if detailed else self.export_trace()
        print(json.dumps(trace_data, indent=2, ensure_ascii=False))

    def get_span_by_type(self, span_type: str) -> List[Dict[str, Any]]:
        """Get all spans of a specific type"""
        return [span for span in self.spans.values() if span["type"] == span_type]

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all spans with errors"""
        return [span for span in self.spans.values() if span.get("error")]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        durations = [span.get("duration_ms", 0)
                     for span in self.spans.values()]

        return {
            "total_duration_ms": sum(durations),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "span_count": len(self.spans),
            "error_count": len(self.get_errors())
        }

    def to_json(self) -> str:
        """Export all trace data as JSON string"""
        trace_data = {
            "trace": self.export_trace(),
            "detailed_trace": self.export_detailed_trace(),
            "performance_stats": self.get_performance_stats(),
            "errors": self.get_errors(),
            "span_types": {
                "llm": self.get_span_by_type("llm"),
                "chain": self.get_span_by_type("chain"),
                "tool": self.get_span_by_type("tool"),
                "retriever": self.get_span_by_type("retriever")
            }
        }
        return json.loads(json.dumps(trace_data, ensure_ascii=False, cls=LangChainEncoder))


# ============ Usage Example ============
#     collector = TraceCollector()

#     # Simulate a chain start
#     chain = your_chain_or_agent(callbacks=[collector])
#     result = chain.invoke({"input": "Your query"})

#     # Get the trace as JSON
#     trace_json = collector.to_json()

#     # return trace_json
#     return trace_json

# ============ Example of implementation ============
# def query_debug(self, user_query: str, user_id: str) -> str:

#     try:
#         message_state = {"messages": [HumanMessage(content=user_query)]}
#         trace_collector = TraceCollector()

#         self.chat_model.invoke(
#             message_state,
#             config={"configurable": {"thread_id": user_id},
#                     "callbacks": [trace_collector]}
#         )

#         return json.dumps(trace_collector.to_json())

#     except Exception as e:
#         logging.error(f"Error in query: {e}")
#         return f"I encountered an error while processing your request: {str(e)}"
