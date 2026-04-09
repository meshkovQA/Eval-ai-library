/* Eval AI Library - API Connector Frontend (v4) */

const state = {
    currentStep: 1,
    activeTab: 'connection',
    activeMetricsCat: 'rag',
    datasetId: null,
    columns: [],
    previewRows: [],
    rowCount: 0,
    apiConfig: {
        base_url: '',
        method: 'POST',
        headers: [{ key: 'Content-Type', value: 'application/json', enabled: true }],
        query_params: {},
        body_template: '',
        timeout_seconds: 60,
        max_retries: 1,
        delay_between_requests_ms: 0,
    },
    testResponse: null,
    responseMapping: {
        actual_output_path: '',
        retrieval_context_path: '',
        tools_called_path: '',
        token_usage_path: '',
        system_prompt_path: '',
    },
    costPer1mTokens: 0,
    columnMapping: {
        input_column: 'input',
        expected_output_column: '',
        context_column: '',
        tools_called_column: '',
        expected_tools_column: '',
        template_variable_map: {},
    },
    selectedMetrics: {},
    evalModel: 'gpt-4o-mini',
    metricsInfo: [],
    testValues: {},
    providers: [],
    _activeProvider: null,
    customLlmConfig: { base_url: '', api_key: '', model_name: '' },
    jobId: null,
    jobPollingInterval: null,
    savedConfigs: [],
    activeConfigId: null,
};

// ---- Init ----
document.addEventListener('DOMContentLoaded', async () => {
    await Promise.all([loadMetrics(), loadConfigs(), loadProviders()]);
    renderStep(1);
    document.querySelectorAll('.step').forEach(el => {
        el.style.opacity = '1';
        el.style.pointerEvents = 'auto';
        el.addEventListener('click', () => {
            saveCurrentStep();
            goToStep(parseInt(el.dataset.step));
        });
    });
    // Close custom dropdowns on outside click
    document.addEventListener('click', e => {
        if (!e.target.closest('.custom-select')) {
            document.querySelectorAll('.custom-select-dropdown.open').forEach(d => d.classList.remove('open'));
            document.querySelectorAll('.custom-select-trigger.open').forEach(t => t.classList.remove('open'));
        }
    });
});

// ---- Navigation ----
function goToStep(step) {
    if (step < 1 || step > 4) return;
    state.currentStep = step;
    updateStepper();
    renderStep(step);
}
function nextStep() {
    saveCurrentStep();
    const err = validateStep(state.currentStep);
    if (err) { showToast(err, 'error'); return; }
    goToStep(state.currentStep + 1);
}
function prevStep() { saveCurrentStep(); goToStep(state.currentStep - 1); }

function validateStep(step) {
    if (step === 1) {
        if (!state.apiConfig.base_url.trim()) return 'Base URL is required';
        if (!state.apiConfig.body_template.trim() && state.apiConfig.method !== 'GET') return 'Request body is required';
        if (Object.values(state.selectedMetrics).filter(v => v.enabled).length === 0) return 'Select at least one metric';
    }
    if (step === 2) {
        if (!state.datasetId) return 'Upload a dataset first';
        if (!state.columnMapping.input_column.trim()) return 'Input column mapping is required';
    }
    if (step === 3) {
        if (!state.responseMapping.actual_output_path.trim()) return 'actual_output mapping is required';
    }
    return null;
}

function renderStepHeader(title, subtitle, nextLabel) {
    let html = `<div class="step-header"><div class="step-header-left">
        <h2 class="step-title">${title}</h2>
        <p class="step-subtitle">${subtitle}</p>
    </div><div class="step-header-right">`;
    if (state.currentStep > 1) {
        html += `<button class="btn" onclick="prevStep()">Back</button>`;
    }
    if (nextLabel) {
        html += `<button class="btn btn-primary" onclick="nextStep()">${nextLabel}</button>`;
    }
    html += `</div></div>`;
    return html;
}

function updateStepper() {
    document.querySelectorAll('.step').forEach(el => {
        const s = parseInt(el.dataset.step);
        el.classList.toggle('active', s === state.currentStep);
    });
}

function saveCurrentStep() {
    if (state.currentStep === 1) saveStep1();
}

function renderStep(step) {
    _jsonNodeId = 0;
    const el = document.getElementById('stepContent');
    switch (step) {
        case 1: el.innerHTML = renderStep1(); initStep1(); break;
        case 2: el.innerHTML = renderStep2(); initStep2(); break;
        case 3: el.innerHTML = renderStep3(); break;
        case 4: el.innerHTML = renderStep4(); break;
    }
    updateStepper();
}

// ---- Custom Select ----
function renderCustomSelect(id, options, selectedValue, placeholder) {
    const sel = options.find(o => o.value === selectedValue);
    const label = sel ? sel.label : (placeholder || 'Select...');
    const cls = sel ? '' : 'style="color:var(--text-muted)"';
    let html = `<div class="custom-select" id="${id}">
        <div class="custom-select-trigger" onclick="toggleDropdown('${id}')" ${cls}>${esc(label)}</div>
        <div class="custom-select-dropdown">`;
    if (placeholder) {
        html += `<div class="custom-select-option${!selectedValue ? ' selected' : ''}" data-value="" onclick="selectOption('${id}','',this)">${esc(placeholder)}</div>`;
    }
    options.forEach(o => {
        const s = o.value === selectedValue ? ' selected' : '';
        html += `<div class="custom-select-option${s}" data-value="${esc(o.value)}" onclick="selectOption('${id}','${esc(o.value)}',this)">${esc(o.label)}</div>`;
    });
    html += '</div></div>';
    return html;
}

function toggleDropdown(id) {
    const el = document.getElementById(id);
    const trigger = el.querySelector('.custom-select-trigger');
    const dd = el.querySelector('.custom-select-dropdown');
    // Close others first
    document.querySelectorAll('.custom-select-dropdown.open').forEach(d => { if (d !== dd) d.classList.remove('open'); });
    document.querySelectorAll('.custom-select-trigger.open').forEach(t => { if (t !== trigger) t.classList.remove('open'); });
    trigger.classList.toggle('open');
    dd.classList.toggle('open');
}

function selectOption(id, value, optEl) {
    const el = document.getElementById(id);
    const trigger = el.querySelector('.custom-select-trigger');
    const dd = el.querySelector('.custom-select-dropdown');
    // Update selection visual
    dd.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
    optEl.classList.add('selected');
    trigger.textContent = optEl.textContent;
    trigger.style.color = value ? '' : 'var(--text-muted)';
    trigger.classList.remove('open');
    dd.classList.remove('open');
    // Dispatch change
    el.dataset.value = value;
    el.dispatchEvent(new CustomEvent('change', { detail: value }));
}

function getSelectValue(id) {
    const el = document.getElementById(id);
    return el ? (el.dataset.value || '') : '';
}

// ---- Tab system ----
function switchTab(tab) {
    saveStep1();
    state.activeTab = tab;
    renderStep(1);
}

function renderTabBar() {
    const metricsCount = Object.values(state.selectedMetrics).filter(v => v.enabled).length;
    const tabs = [
        { id: 'connection', label: 'Connection', extra: '' },
        { id: 'mapping', label: 'Response', extra: state.testResponse?.response_body ? '<span class="tab-dot"></span>' : '' },
        { id: 'columns', label: 'Columns', extra: '' },
        { id: 'metrics', label: 'Metrics', extra: metricsCount ? `<span class="tab-badge">${metricsCount}</span>` : '' },
        { id: 'settings', label: 'Settings', extra: '' },
    ];
    let html = '<div class="tab-bar">';
    tabs.forEach(t => {
        const cls = t.id === state.activeTab ? 'tab-item active' : 'tab-item';
        html += `<button class="${cls}" onclick="switchTab('${t.id}')">${t.label}${t.extra}</button>`;
    });
    html += '</div>';
    return html;
}

// ==== STEP 1: Project Settings with Tabs ====

function renderStep1() {
    let html = renderStepHeader('Project Settings', 'Configure your API connector', 'Next: Dataset');

    // Config toolbar
    html += renderConfigToolbar();

    // Eval Model selector — always visible at the top
    html += renderEvalModelSection();

    html += renderTabBar();
    html += '<div class="tab-content">';

    switch (state.activeTab) {
        case 'connection': html += renderTabConnection(); break;
        case 'mapping': html += renderTabMapping(); break;
        case 'columns': html += renderTabColumns(); break;
        case 'metrics': html += renderTabMetrics(); break;
        case 'settings': html += renderTabSettings(); break;
    }

    html += '</div>';

    return html;
}

// Parse a stored eval model string ("openai:gpt-4o", "gpt-4o-mini") into
// {providerId, model}. Bare strings (no colon) are assumed to be openai.
function parseEvalModel(value) {
    if (!value) return { providerId: '', model: '' };
    const idx = value.indexOf(':');
    if (idx === -1) return { providerId: 'openai', model: value };
    return { providerId: value.slice(0, idx), model: value.slice(idx + 1) };
}

// Build the canonical "provider:model" string used by the backend.
// OpenAI is the default and can be passed bare; everything else needs the prefix.
function formatEvalModel(providerId, model) {
    if (!providerId || !model) return '';
    if (providerId === 'openai') return model;
    return `${providerId}:${model}`;
}

function renderEvalModelSection() {
    const configuredProviders = state.providers.filter(p => p.has_key || (p.key_optional && Object.values(p.extra_configured || {}).some(v => v)));

    // Resolve current provider/model from state.evalModel. If the stored
    // provider isn't configured anymore, fall back to the first configured one.
    let { providerId, model } = parseEvalModel(state.evalModel);
    if (configuredProviders.length && !configuredProviders.find(p => p.id === providerId)) {
        providerId = configuredProviders[0].id;
        model = '';
    }
    const activeProvider = configuredProviders.find(p => p.id === providerId);
    if (activeProvider && activeProvider.models.length && !activeProvider.models.includes(model)) {
        model = activeProvider.models[0];
    }
    // Sync back so subsequent saves see the resolved values
    state.evalModel = formatEvalModel(providerId, model);

    const providerOpts = configuredProviders.map(p => ({
        value: p.id,
        label: `${p.name} (${p.models.length})`,
    }));
    const modelOpts = (activeProvider?.models || []).map(m => ({ value: m, label: m }));

    let html = `<div class="section-card" style="margin-bottom:16px;display:flex;align-items:center;gap:12px;padding:12px 18px;flex-wrap:wrap">
        <div style="flex-shrink:0">
            <div style="font-size:0.75em;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.3px">Eval Model</div>
        </div>`;

    if (configuredProviders.length) {
        html += `<div style="flex:0 0 200px">${renderCustomSelect('evalProviderSelect', providerOpts, providerId, 'Provider...')}</div>`;
        html += `<div style="flex:1 1 260px;min-width:220px;max-width:380px">${renderCustomSelect('evalModelSelect', modelOpts, model, modelOpts.length ? 'Model...' : 'No models')}</div>`;
    } else {
        html += `<div style="flex:1"><span style="color:var(--text-muted);font-size:0.85em">Configure a provider in Settings tab first</span></div>`;
    }

    html += `<div style="margin-left:auto;display:flex;align-items:center;gap:8px">`;
    const configuredCount = configuredProviders.length;
    const totalCount = state.providers.length;
    html += `<span style="font-size:0.75em;color:var(--text-muted)">${configuredCount}/${totalCount} providers</span>`;
    if (!configuredCount) {
        html += `<button class="btn btn-sm btn-secondary" onclick="switchTab('settings')" style="font-size:0.75em">Configure</button>`;
    }
    html += `</div></div>`;
    return html;
}

function renderConfigToolbar() {
    const active = state.savedConfigs.find(c => c.id === state.activeConfigId);
    let html = '<div class="config-toolbar">';

    if (active) {
        html += `<span class="config-active-name">${esc(active.name)}</span>`;
        html += `<button class="btn btn-sm btn-primary" onclick="saveProjectConfig()" title="Save changes">Save</button>`;
        html += `<button class="btn btn-sm" onclick="saveProjectConfigAs()" title="Save as new">Save As</button>`;
        html += `<button class="btn btn-sm" onclick="resetProject()" title="New project">New</button>`;
        html += `<button class="btn btn-sm btn-danger" onclick="deleteCurrentConfig()" title="Delete">Delete</button>`;
    } else {
        html += `<span class="config-active-name" style="color:var(--text-muted)">Unsaved project</span>`;
        html += `<button class="btn btn-sm btn-primary" onclick="saveProjectConfig()">Save Project</button>`;
    }

    if (state.savedConfigs.length) {
        html += `<span style="margin-left:auto"></span>`;
        const configOpts = state.savedConfigs.map(c => ({ value: c.id, label: c.name }));
        html += `<div style="max-width:240px;margin-left:auto">${renderCustomSelect('configSelect', configOpts, state.activeConfigId || '', 'Load project...')}</div>`;
    }

    html += '</div>';
    return html;
}

// ---- Tab: Connection ----
function renderTabConnection() {
    const a = state.apiConfig;
    let html = '';

    // API Endpoint
    const methodOptions = [
        { value: 'GET', label: 'GET' },
        { value: 'POST', label: 'POST' },
        { value: 'PUT', label: 'PUT' },
    ];
    html += `<div class="section-card">
        <div class="section-card-title">API Endpoint</div>
        <div class="form-row" style="grid-template-columns: 120px 1fr">
            ${renderCustomSelect('apiMethod', methodOptions, a.method, '')}
            <input class="form-input" id="apiUrl" placeholder="https://api.example.com/v1/chat/completions" value="${esc(a.base_url)}">
        </div>
    </div>`;

    // Headers
    html += `<div class="section-card">
        <div class="section-card-title">Headers</div>
        <div class="kv-editor" id="headersEditor">`;
    a.headers.forEach((h, i) => {
        html += `<div class="kv-row">
            <input class="form-input" placeholder="Key" value="${esc(h.key)}" data-prefix="header" data-idx="${i}" data-field="key">
            <input class="form-input" placeholder="Value" value="${esc(h.value)}" data-prefix="header" data-idx="${i}" data-field="value">
            <button class="kv-remove" onclick="removeHeader(${i})">x</button>
        </div>`;
    });
    html += `</div><button class="kv-add" onclick="addHeader()">+ Add header</button></div>`;

    // Query params
    const qpEntries = Object.entries(a.query_params);
    html += `<div class="section-card">
        <div class="section-card-title">Query Parameters</div>
        <p style="font-size:0.8em;color:var(--text-muted);margin-bottom:8px">Added to the URL as <code style="background:var(--accent-light);padding:1px 4px;border-radius:3px;color:var(--accent)">?key=value</code>. Supports <code style="background:var(--accent-light);padding:1px 4px;border-radius:3px;color:var(--accent)">{{variables}}</code>.</p>
        <div class="kv-editor" id="queryParamsEditor">`;
    qpEntries.forEach(([k, v], i) => {
        html += `<div class="kv-row">
            <input class="form-input" placeholder="Key" value="${esc(k)}" data-prefix="qp" data-idx="${i}" data-field="key">
            <input class="form-input" placeholder="Value" value="${esc(v)}" data-prefix="qp" data-idx="${i}" data-field="value">
            <button class="kv-remove" onclick="removeQueryParam(${i})">x</button>
        </div>`;
    });
    html += `</div><button class="kv-add" onclick="addQueryParam()">+ Add parameter</button></div>`;

    // Body template
    html += `<div class="section-card">
        <div class="section-card-title">Request Body Template</div>
        <p style="font-size:0.8em;color:var(--text-muted);margin-bottom:8px">Use <code style="background:var(--accent-light);padding:1px 4px;border-radius:3px;color:var(--accent)">{{column_name}}</code> placeholders. Supported in URL, headers, query params, and body.</p>
        <textarea class="form-textarea" id="apiBody" rows="8" placeholder='{"messages":[{"role":"user","content":"{{input}}"}]}'>${esc(a.body_template)}</textarea>
    </div>`;

    // Test connection with variables
    html += `<div class="section-card">
        <div class="section-card-title">Test Connection</div>
        <p style="font-size:0.8em;color:var(--text-muted);margin-bottom:12px">Send a test request. The response will be available in the <strong>Mapping</strong> tab.</p>`;

    const tplVars = getTemplateVars();
    if (tplVars.length) {
        html += `<div style="margin-bottom:12px">
            <div style="font-size:0.75em;color:var(--text-secondary);font-weight:600;text-transform:uppercase;letter-spacing:0.3px;margin-bottom:8px">Test Values for Variables</div>`;
        if (state.previewRows.length) {
            html += `<p style="font-size:0.75em;color:var(--text-muted);margin-bottom:8px">Values from dataset row 1. Override below if needed.</p>`;
        } else {
            html += `<p style="font-size:0.75em;color:var(--text-muted);margin-bottom:8px">No dataset loaded. Enter test values to send a request.</p>`;
        }
        tplVars.forEach(v => {
            const col = state.columnMapping.template_variable_map[v] || v;
            const datasetVal = state.previewRows[0]?.[col] || '';
            const testVal = state.testValues[v] ?? (datasetVal ? String(datasetVal) : '');
            html += `<div class="mapping-field" style="margin-bottom:6px">
                <div class="mapping-field-label" style="font-size:0.8em"><code style="color:var(--accent)">{{${esc(v)}}}</code></div>
                <input class="form-input" id="testval_${esc(v)}" value="${esc(testVal)}" placeholder="Enter test value..." style="font-size:0.8em">
            </div>`;
        });
        html += '</div>';
    }

    html += `<button class="btn btn-secondary" onclick="testConnection()" id="testBtn">Send Test Request</button>
        <div id="testResult"></div>`;

    if (state.testResponse) {
        if (state.testResponse.error) {
            html += `<div class="error-item" style="margin-top:10px">${esc(state.testResponse.error)}</div>`;
        } else if (state.testResponse.response_body) {
            const data = state.testResponse;
            const isOk = data.status_code >= 200 && data.status_code < 400;
            html += `<div style="margin-top:12px">
                <div class="response-status">
                    <span class="status-code ${isOk ? 'success' : 'error'}">${data.status_code}</span>
                    <span style="color:var(--text-muted)">${data.elapsed_ms}ms</span>
                </div>
                ${data.sent_body ? `<details style="margin:8px 0"><summary style="font-size:0.8em;color:var(--text-muted);cursor:pointer">Sent body</summary><pre style="font-size:0.8em;background:var(--bg-secondary);padding:8px;margin-top:4px;overflow-x:auto;white-space:pre-wrap">${esc(data.sent_body)}</pre></details>` : ''}
                <div class="json-tree" style="max-height:200px">${renderJsonTree(data.response_body, '', 0)}</div>
                <p style="font-size:0.8em;color:var(--accent);margin-top:8px;cursor:pointer" onclick="switchTab('mapping')">Configure response mapping &rarr;</p>
            </div>`;
        }
    }

    html += '</div>';
    return html;
}

// ---- Tab: Mapping ----
function renderTabMapping() {
    let html = '';

    // Response mapping
    html += `<div class="section-card">
        <div class="section-card-title">Response Mapping</div>
        <p style="font-size:0.8em;color:var(--text-muted);margin-bottom:12px">JSONPath to extract values from API responses. Click on values in the JSON tree to auto-fill.</p>
        <div class="mapping-field">
            <div class="mapping-field-label">actual_output <span class="required">*</span><br><span class="field-hint">Main text response from the API</span></div>
            <input class="form-input" id="mapActualOutput" placeholder="choices[0].message.content" value="${esc(state.responseMapping.actual_output_path)}" oninput="updateMappingPreview()">
            <div class="mapping-live-preview" id="preview_actual_output"></div>
        </div>
        <div class="mapping-field">
            <div class="mapping-field-label">retrieval_context <span class="optional">(optional)</span><br><span class="field-hint">Sources/chunks used to generate the answer (for RAG metrics)</span></div>
            <input class="form-input" id="mapRetrievalContext" placeholder="sources[*].content" value="${esc(state.responseMapping.retrieval_context_path)}" oninput="updateMappingPreview()">
            <div class="mapping-live-preview" id="preview_retrieval_context"></div>
        </div>
        <div class="mapping-field">
            <div class="mapping-field-label">tools_called <span class="optional">(optional)</span><br><span class="field-hint">Tools/functions invoked by the agent (for Agent metrics)</span></div>
            <input class="form-input" id="mapToolsCalled" placeholder="tool_calls[*].function.name" value="${esc(state.responseMapping.tools_called_path || '')}" oninput="updateMappingPreview()">
            <div class="mapping-live-preview" id="preview_tools_called"></div>
        </div>
        <div class="mapping-field">
            <div class="mapping-field-label">token_usage <span class="optional">(optional)</span><br><span class="field-hint">Total tokens used (for cost calculation)</span></div>
            <input class="form-input" id="mapTokenUsage" placeholder="usage.total_tokens" value="${esc(state.responseMapping.token_usage_path || '')}" oninput="updateMappingPreview()">
            <div class="mapping-live-preview" id="preview_token_usage"></div>
        </div>
        <div class="mapping-field">
            <div class="mapping-field-label">system_prompt <span class="optional">(optional)</span><br><span class="field-hint">System prompt used by the AI</span></div>
            <input class="form-input" id="mapSystemPrompt" placeholder="system_prompt" value="${esc(state.responseMapping.system_prompt_path || '')}" oninput="updateMappingPreview()">
            <div class="mapping-live-preview" id="preview_system_prompt"></div>
        </div>
    </div>`;

    // JSON tree from test response
    if (state.testResponse?.response_body) {
        html += `<div class="section-card">
            <div class="section-card-title">API Response — click a value to copy its path</div>
            <div class="json-tree" id="jsonTree">${renderJsonTree(state.testResponse.response_body, '', 0)}</div>
        </div>`;

    } else {
        html += `<div class="section-card" style="text-align:center;padding:40px;color:var(--text-muted)">
            <p style="margin-bottom:8px">No test response yet</p>
            <p style="font-size:0.85em">Send a test request in the <strong style="cursor:pointer;color:var(--accent)" onclick="switchTab('connection')">Connection</strong> tab first</p>
        </div>`;
    }

    return html;
}

// ---- Tab: Columns (dataset column mapping) ----
function renderTabColumns() {
    let html = '';

    html += `<div class="section-card">
        <div class="section-card-title">Dataset Column Mapping</div>
        <p style="font-size:0.8em;color:var(--text-muted);margin-bottom:12px">Map dataset columns to EvalTestCase fields</p>`;

    const fields = [
        { id: 'input_column', label: 'input', required: true, val: state.columnMapping.input_column, hint: 'User query / question' },
        { id: 'expected_output_column', label: 'expected_output', required: false, val: state.columnMapping.expected_output_column, hint: 'Ground truth answer' },
        { id: 'expected_tools_column', label: 'expected_tools', required: false, val: state.columnMapping.expected_tools_column, hint: 'Expected tools for Agent metrics' },
    ];
    fields.forEach(f => {
        const req = f.required ? '<span class="required">*</span>' : '<span class="optional">(optional)</span>';
        const hint = f.hint ? `<br><span class="field-hint">${f.hint}</span>` : '';
        html += `<div class="mapping-field">
            <div class="mapping-field-label">${f.label} ${req}${hint}</div>
            <input class="form-input" id="col_${f.id}" placeholder="${f.label}" value="${esc(f.val)}">
        </div>`;
    });

    // Template variable mapping (exclude vars that match standard column mapping fields)
    const standardCols = new Set([
        state.columnMapping.input_column,
        state.columnMapping.expected_output_column,
        state.columnMapping.expected_tools_column,
    ].filter(Boolean));
    const tplVars = getTemplateVars().filter(v => !standardCols.has(v));
    if (tplVars.length > 0) {
        html += `<div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border)">
            <div style="font-weight:600;font-size:0.9em;margin-bottom:4px">Template Variables</div>
            <p style="font-size:0.8em;color:var(--text-muted);margin-bottom:12px">Map <code>{{variable}}</code> placeholders from your request to dataset columns</p>`;
        const datasetCols = state.datasetColumns || [];
        tplVars.forEach(v => {
            const mapped = state.columnMapping.template_variable_map[v] || v;
            html += `<div class="mapping-field">
                <div class="mapping-field-label"><code>{{${esc(v)}}}</code><br><span class="field-hint">Dataset column to use</span></div>`;
            if (datasetCols.length > 0) {
                const colOpts = datasetCols.map(c => ({ value: c, label: c }));
                if (mapped && !datasetCols.includes(mapped)) {
                    colOpts.push({ value: mapped, label: mapped });
                }
                html += renderCustomSelect(`tplmap_${v}`, colOpts, mapped, '');
            } else {
                html += `<input class="form-input" id="tplmap_${esc(v)}" placeholder="${v}" value="${esc(mapped)}">`;
            }
            html += `</div>`;
        });
        html += '</div>';
    }

    html += '</div>';

    return html;
}

// ---- Tab: Metrics (card grid with sub-tabs) ----
function switchMetricsCat(cat) {
    saveMetricParams();
    state.activeMetricsCat = cat;
    renderStep(1);
}

function renderTabMetrics() {
    let html = '';
    const categories = { rag: 'RAG', agent: 'Agent', security: 'Security', deterministic: 'Deterministic', vector: 'Vector' };

    // Sub-tab bar
    html += '<div class="metrics-subtabs">';
    for (const [cat, label] of Object.entries(categories)) {
        const metrics = state.metricsInfo.filter(m => m.category === cat);
        if (!metrics.length) continue;
        const count = metrics.filter(m => state.selectedMetrics[m.name]?.enabled).length;
        const cls = cat === state.activeMetricsCat ? 'metrics-subtab active' : 'metrics-subtab';
        html += `<button class="${cls}" onclick="switchMetricsCat('${cat}')">${label}${count ? ` <span class="tab-badge">${count}</span>` : ''}</button>`;
    }
    html += '</div>';

    // Render cards for active category
    const metrics = state.metricsInfo.filter(m => m.category === state.activeMetricsCat);
    html += '<div class="metrics-grid-cards">';
    metrics.forEach(m => {
        const sel = state.selectedMetrics[m.name] || { enabled: false, params: {} };
        const selectedCls = sel.enabled ? ' selected' : '';

        html += `<div class="metric-card-item${selectedCls}" onclick="toggleMetricCard('${m.name}', event)">
            <div class="metric-card-header">
                <div class="metric-card-check">\u2713</div>
                <div class="metric-card-name">${m.name}</div>
            </div>
            <div class="metric-card-desc">${esc(m.description)}</div>
            <div class="metric-card-tags">
                ${m.required_fields.map(f => `<span class="metric-card-tag">${f}</span>`).join('')}
            </div>`;

        if (m.params.length) {
            html += '<div class="metric-card-params" onclick="event.stopPropagation()">';
            m.params.forEach(p => {
                const val = sel.params[p.name] != null ? sel.params[p.name] : p.default;
                html += `<div class="metric-param"><label>${p.name}</label>`;
                if (p.type === 'select') {
                    const csId = `mcs_${m.name}_${p.name}`;
                    const opts = (p.options || []).map(o => ({ value: o, label: o }));
                    html += renderCustomSelect(csId, opts, val != null ? String(val) : '', '');
                    html += `<input type="hidden" class="metric-cs-hidden" data-metric="${m.name}" data-param="${p.name}" data-cs-id="${csId}" value="${esc(val != null ? String(val) : '')}">`;
                } else if (p.type === 'bool') {
                    const csId = `mcs_${m.name}_${p.name}`;
                    const boolOpts = [{ value: 'true', label: 'true' }, { value: 'false', label: 'false' }];
                    html += renderCustomSelect(csId, boolOpts, val != null ? String(val) : 'false', '');
                    html += `<input type="hidden" class="metric-cs-hidden" data-metric="${m.name}" data-param="${p.name}" data-cs-id="${csId}" value="${val ? 'true' : 'false'}">`;
                } else if (p.type === 'text' || p.type === 'string') {
                    html += `<input type="text" value="${esc(val || '')}" data-metric="${m.name}" data-param="${p.name}">`;
                } else if (p.type === 'list') {
                    html += `<input type="text" value="${esc(val ? (Array.isArray(val) ? val.join(', ') : '') : '')}" data-metric="${m.name}" data-param="${p.name}" placeholder="comma-separated">`;
                } else {
                    const step = p.type === 'int' ? '1' : '0.05';
                    html += `<input type="number" step="${step}" min="${p.min||0}" max="${p.max||''}" value="${val != null ? val : ''}" data-metric="${m.name}" data-param="${p.name}">`;
                }
                html += '</div>';
            });
            html += '</div>';
        }

        html += '</div>';
    });
    html += '</div>';

    return html;
}

function toggleMetricCard(name, event) {
    if (event.target.closest('.metric-card-params')) return;
    saveMetricParams();
    if (!state.selectedMetrics[name]) state.selectedMetrics[name] = { enabled: false, params: {} };
    state.selectedMetrics[name].enabled = !state.selectedMetrics[name].enabled;
    renderStep(1);
}

// ---- Tab: Settings ----
function renderTabSettings() {
    const a = state.apiConfig;
    let html = '';

    // LLM Providers — dropdown selector + configuration panel for the chosen one
    const isProviderReady = (p) => p.has_key || (p.key_optional && Object.values(p.extra_configured || {}).some(v => v));

    // Pick a default active provider on first render: prefer one that's already configured
    if (!state._activeProvider && state.providers.length) {
        const firstReady = state.providers.find(isProviderReady);
        state._activeProvider = (firstReady || state.providers[0]).id;
    }

    const providerOpts = state.providers.map(p => ({
        value: p.id,
        label: `${isProviderReady(p) ? '\u2713 ' : '\u25CB '}${p.name}`,
    }));
    const configuredCount = state.providers.filter(isProviderReady).length;

    html += `<div class="section-card">
        <div class="section-card-title">LLM Providers</div>
        <p style="font-size:0.8em;color:var(--text-muted);margin-bottom:12px">Pick a provider from the list and configure its API key. Keys are stored locally. ${configuredCount}/${state.providers.length} configured.</p>
        <div style="max-width:340px;margin-bottom:14px">
            ${renderCustomSelect('settingsProviderSelect', providerOpts, state._activeProvider || '', 'Select provider...')}
        </div>`;

    // Active provider config panel
    const activeP = state.providers.find(p => p.id === state._activeProvider);
    if (activeP) {
        const isConfigured = activeP.has_key || (activeP.key_optional && Object.values(activeP.extra_configured || {}).some(v => v));

        if (activeP.is_custom_llm) {
            // Custom LLM configuration panel
            const clCfg = activeP.custom_llm_config || {};
            html += `<div style="border:1px solid var(--border);border-radius:var(--radius);padding:14px;background:var(--bg-secondary)">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                    <span style="font-weight:600;font-size:0.9em;color:var(--text)">${esc(activeP.name)}</span>
                    <span style="font-size:0.75em;color:${isConfigured ? 'var(--success)' : 'var(--text-muted)'}">${isConfigured ? '&#10003; Configured' : 'Not configured'}</span>
                </div>
                <p style="font-size:0.78em;color:var(--text-muted);margin-bottom:10px">Connect any OpenAI-compatible API endpoint (e.g. LiteLLM, vLLM, LocalAI, custom server).</p>
                <div style="display:flex;flex-direction:column;gap:8px">
                    <div>
                        <label style="font-size:0.75em;font-weight:600;color:var(--text-secondary);display:block;margin-bottom:3px">Base URL *</label>
                        <input class="form-input" id="customLlmBaseUrl" type="text"
                            placeholder="https://your-server.com/v1"
                            value="${esc(clCfg.base_url || '')}"
                            style="font-size:0.8em">
                    </div>
                    <div>
                        <label style="font-size:0.75em;font-weight:600;color:var(--text-secondary);display:block;margin-bottom:3px">API Key</label>
                        <input class="form-input" id="customLlmApiKey" type="password"
                            placeholder="${clCfg.api_key ? '••••••••' : 'Optional — leave empty if not required'}"
                            style="font-size:0.8em">
                    </div>
                    <div>
                        <label style="font-size:0.75em;font-weight:600;color:var(--text-secondary);display:block;margin-bottom:3px">Model name *</label>
                        <input class="form-input" id="customLlmModelName" type="text"
                            placeholder="e.g. gpt-4o-mini, my-model, llama3"
                            value="${esc(clCfg.model_name || '')}"
                            style="font-size:0.8em">
                    </div>
                    <button class="btn btn-sm btn-primary" onclick="saveCustomLlmConfig()" style="align-self:flex-start;margin-top:4px">Save Configuration</button>
                </div>
            </div>`;
        } else {
            // Standard provider config panel
            html += `<div style="border:1px solid var(--border);border-radius:var(--radius);padding:14px;background:var(--bg-secondary)">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                    <span style="font-weight:600;font-size:0.9em;color:var(--text)">${esc(activeP.name)}</span>
                    <span style="font-size:0.75em;color:${isConfigured ? 'var(--success)' : 'var(--text-muted)'}">${isConfigured ? '&#10003; Configured' : 'Not configured'}</span>
                </div>
                <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px">
                    <input class="form-input" id="key_${esc(activeP.env_var)}" type="password"
                        placeholder="${activeP.has_key ? '••••••••••••' : activeP.env_var}"
                        style="font-size:0.8em;flex:1">
                    <button class="btn btn-sm btn-primary" onclick="saveApiKey('${esc(activeP.env_var)}', document.getElementById('key_${esc(activeP.env_var)}').value)" style="white-space:nowrap">Save</button>
                    ${activeP.has_key ? `<button class="btn btn-sm btn-danger" onclick="if(confirm('Remove key?')) deleteApiKey('${esc(activeP.env_var)}')" style="padding:4px 8px" title="Remove key">&#10005;</button>` : ''}
                </div>`;

            if (activeP.extra_vars?.length) {
                activeP.extra_vars.forEach(ev => {
                    html += `<div style="display:flex;gap:6px;align-items:center;margin-top:6px">
                        <input class="form-input" id="key_${esc(ev)}" type="text"
                            placeholder="${activeP.extra_configured?.[ev] ? '••••••••' : ev}"
                            style="font-size:0.8em;flex:1">
                        <button class="btn btn-sm" onclick="saveApiKey('${esc(ev)}', document.getElementById('key_${esc(ev)}').value)" style="white-space:nowrap">Save</button>
                    </div>`;
                });
            }

            if (activeP.models.length) {
                html += `<div style="margin-top:8px;font-size:0.75em;color:var(--text-muted)">
                    ${activeP.models.length} models available &mdash; pick one when launching an evaluation.
                </div>`;
            }

            html += `</div>`;
        }
    }

    html += `</div>`;

    // Request settings
    html += `<div class="section-card">
        <div class="section-card-title">Request Settings</div>
        <div class="form-row" style="grid-template-columns: 1fr 1fr">
            <div class="form-group">
                <label class="form-label">Timeout (seconds)</label>
                <input type="number" class="form-input" id="apiTimeout" value="${a.timeout_seconds}" min="1">
            </div>
            <div class="form-group">
                <label class="form-label">Max Retries</label>
                <input type="number" class="form-input" id="apiRetries" value="${a.max_retries}" min="1">
            </div>
        </div>
        <div class="form-row" style="grid-template-columns: 1fr 1fr">
            <div class="form-group">
                <label class="form-label">Delay Between Requests (ms)</label>
                <input type="number" class="form-input" id="apiDelay" value="${a.delay_between_requests_ms}" min="0">
            </div>
            <div class="form-group"></div>
        </div>
    </div>`;

    // Cost settings
    html += `<div class="section-card">
        <div class="section-card-title">Cost Tracking</div>
        <p style="font-size:0.8em;color:var(--text-muted);margin-bottom:12px">Set the cost per 1M tokens for your AI project to calculate AI costs. Leave 0 to disable.</p>
        <div class="form-row" style="grid-template-columns: 1fr 1fr">
            <div class="form-group">
                <label class="form-label">Cost per 1M tokens ($)</label>
                <input type="number" class="form-input" id="costPer1m" value="${state.costPer1mTokens || 0}" min="0" step="0.001">
            </div>
            <div class="form-group"></div>
        </div>
    </div>`;

    return html;
}

function onEvalModelChange() {
    // Recompose state.evalModel from the two cascade selects.
    const providerId = getSelectValue('evalProviderSelect') || parseEvalModel(state.evalModel).providerId;
    const model = getSelectValue('evalModelSelect') || parseEvalModel(state.evalModel).model;
    if (providerId && model) state.evalModel = formatEvalModel(providerId, model);
}

function initStep1() {
    ['mapActualOutput', 'mapRetrievalContext', 'mapToolsCalled', 'mapTokenUsage', 'mapSystemPrompt'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => { saveResponseMappingFields(); updateMappingPreview(); });
    });
    updateMappingPreview();

    // Custom select hooks
    const configSel = document.getElementById('configSelect');
    if (configSel) configSel.addEventListener('change', () => loadSelectedConfig());
    const methodSel = document.getElementById('apiMethod');
    if (methodSel) methodSel.addEventListener('change', () => {});

    // Settings tab: provider dropdown swaps the configuration panel below
    const settingsProvSel = document.getElementById('settingsProviderSelect');
    if (settingsProvSel) settingsProvSel.addEventListener('change', (e) => {
        state._activeProvider = e.detail || null;
        renderStep(1);
    });

    // Eval-model cascade: changing the provider re-renders so the model
    // dropdown is repopulated; changing the model just updates state.
    const evalProvSel = document.getElementById('evalProviderSelect');
    if (evalProvSel) evalProvSel.addEventListener('change', (e) => {
        const newPid = e.detail;
        const provider = state.providers.find(p => p.id === newPid);
        const firstModel = provider?.models?.[0] || '';
        state.evalModel = formatEvalModel(newPid, firstModel);
        renderStep(1);
    });
    const evalSel = document.getElementById('evalModelSelect');
    if (evalSel) evalSel.addEventListener('change', (e) => {
        const providerId = getSelectValue('evalProviderSelect') || parseEvalModel(state.evalModel).providerId;
        state.evalModel = formatEvalModel(providerId, e.detail);
    });

    // Bind custom-select dropdowns in metric params to their hidden inputs
    document.querySelectorAll('.metric-cs-hidden').forEach(hidden => {
        const csEl = document.getElementById(hidden.dataset.csId);
        if (csEl) {
            csEl.addEventListener('change', e => { hidden.value = e.detail; });
        }
    });
}

function saveResponseMappingFields() {
    const ao = document.getElementById('mapActualOutput');
    const rc = document.getElementById('mapRetrievalContext');
    const tc = document.getElementById('mapToolsCalled');
    const tu = document.getElementById('mapTokenUsage');
    const sp = document.getElementById('mapSystemPrompt');
    if (ao) state.responseMapping.actual_output_path = ao.value;
    if (rc) state.responseMapping.retrieval_context_path = rc.value;
    if (tc) state.responseMapping.tools_called_path = tc.value;
    if (tu) state.responseMapping.token_usage_path = tu.value;
    if (sp) state.responseMapping.system_prompt_path = sp.value;
}

function saveStep1() {
    const a = state.apiConfig;
    a.base_url = document.getElementById('apiUrl')?.value ?? a.base_url;
    const methodVal = getSelectValue('apiMethod');
    if (methodVal) a.method = methodVal;
    a.body_template = document.getElementById('apiBody')?.value ?? a.body_template;

    const headerInputs = document.querySelectorAll('[data-prefix="header"]');
    if (headerInputs.length) {
        const headers = {};
        headerInputs.forEach(input => {
            const idx = parseInt(input.dataset.idx);
            if (!headers[idx]) headers[idx] = { key: '', value: '', enabled: true };
            headers[idx][input.dataset.field] = input.value;
        });
        a.headers = Object.values(headers).filter(h => h.key || h.value);
    }

    // Query params
    const qpInputs = document.querySelectorAll('[data-prefix="qp"]');
    if (qpInputs.length) {
        const qpPairs = {};
        qpInputs.forEach(input => {
            const idx = parseInt(input.dataset.idx);
            if (!qpPairs[idx]) qpPairs[idx] = { key: '', value: '' };
            qpPairs[idx][input.dataset.field] = input.value;
        });
        a.query_params = {};
        Object.values(qpPairs).forEach(p => { if (p.key) a.query_params[p.key] = p.value; });
    }

    // Test values
    getTemplateVars().forEach(v => {
        const el = document.getElementById(`testval_${v}`);
        if (el) state.testValues[v] = el.value;
    });

    saveResponseMappingFields();
    state.columnMapping.input_column = document.getElementById('col_input_column')?.value ?? state.columnMapping.input_column;
    state.columnMapping.expected_output_column = document.getElementById('col_expected_output_column')?.value ?? state.columnMapping.expected_output_column;
    state.columnMapping.context_column = document.getElementById('col_context_column')?.value ?? state.columnMapping.context_column;
    state.columnMapping.tools_called_column = document.getElementById('col_tools_called_column')?.value ?? state.columnMapping.tools_called_column;
    state.columnMapping.expected_tools_column = document.getElementById('col_expected_tools_column')?.value ?? state.columnMapping.expected_tools_column;

    // Save template variable mappings
    getTemplateVars().forEach(v => {
        const el = document.getElementById(`tplmap_${v}`);
        if (!el) return;
        // Custom select uses dataset.value, input uses .value
        const val = el.classList.contains('custom-select') ? (el.dataset.value || '') : el.value;
        if (val) state.columnMapping.template_variable_map[v] = val;
    });

    a.timeout_seconds = parseInt(document.getElementById('apiTimeout')?.value) || a.timeout_seconds;
    a.max_retries = parseInt(document.getElementById('apiRetries')?.value) || a.max_retries;
    a.delay_between_requests_ms = parseInt(document.getElementById('apiDelay')?.value) || a.delay_between_requests_ms;
    // Compose the canonical "provider:model" from the cascade selects.
    const provVal = getSelectValue('evalProviderSelect');
    const modelVal = getSelectValue('evalModelSelect');
    if (provVal && modelVal) state.evalModel = formatEvalModel(provVal, modelVal);
    const costEl = document.getElementById('costPer1m');
    if (costEl) state.costPer1mTokens = parseFloat(costEl.value) || 0;

    saveMetricParams();
}

function addHeader() {
    saveStep1();
    state.apiConfig.headers.push({ key: '', value: '', enabled: true });
    renderStep(1);
}

function removeHeader(idx) {
    saveStep1();
    state.apiConfig.headers.splice(idx, 1);
    renderStep(1);
}

function addQueryParam() {
    saveStep1();
    state.apiConfig.query_params[''] = '';
    renderStep(1);
}

function removeQueryParam(idx) {
    saveStep1();
    const entries = Object.entries(state.apiConfig.query_params);
    entries.splice(idx, 1);
    state.apiConfig.query_params = Object.fromEntries(entries);
    renderStep(1);
}

function buildTestSampleRow() {
    const sampleRow = {};
    getTemplateVars().forEach(v => {
        const col = state.columnMapping.template_variable_map[v] || v;
        const datasetVal = state.previewRows[0]?.[col];
        const testVal = state.testValues[v];
        if (testVal) {
            // Try to parse JSON to preserve arrays/objects from dataset
            try {
                const parsed = JSON.parse(testVal);
                if (typeof parsed === 'object' && parsed !== null) {
                    sampleRow[col] = parsed;
                    return;
                }
            } catch {}
            sampleRow[col] = testVal;
        } else if (datasetVal !== undefined) {
            sampleRow[col] = datasetVal;
        }
    });
    return sampleRow;
}

async function testConnection() {
    saveStep1();
    const a = state.apiConfig;
    if (!a.base_url) { alert('Please set a URL first'); return; }

    const sampleRow = buildTestSampleRow();

    const vars = getTemplateVars();
    const varMap = {};
    vars.forEach(v => {
        varMap[v] = state.columnMapping.template_variable_map[v] || v;
    });

    const body = {
        base_url: a.base_url,
        method: a.method,
        headers: a.headers.filter(h => h.key && h.enabled),
        query_params: a.query_params,
        body_template: a.body_template,
        timeout_seconds: a.timeout_seconds,
        sample_row: sampleRow,
        variable_map: varMap,
    };

    const btn = document.getElementById('testBtn');
    const target = document.getElementById('testResult');
    if (btn) { btn.disabled = true; btn.textContent = 'Sending...'; }

    try {
        const resp = await fetch('/api/connector/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        state.testResponse = await resp.json();
        renderStep(1);
    } catch (e) {
        if (target) target.innerHTML = `<div class="error-item" style="margin-top:10px">Error: ${esc(e.message)}</div>`;
        if (btn) { btn.disabled = false; btn.textContent = 'Send Test Request'; }
    }
}

// ==== STEP 2: Dataset ====

function renderStep2() {
    let html = renderStepHeader('Dataset', 'Upload your test data or download a template', 'Next: Test');

    const reqCols = getRequiredColumns();
    html += `<div class="section-card">
        <div class="section-card-title">Required Columns</div>
        <p style="font-size:0.85em;color:var(--text-secondary);margin-bottom:12px">Based on your settings, your dataset needs these columns:</p>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px">`;
    reqCols.forEach(c => {
        html += `<span style="padding:3px 10px;background:var(--accent-light);color:var(--accent);font-size:0.8em;font-family:monospace">${esc(c.name)}${c.required ? ' *' : ''}</span>`;
    });
    html += `</div>
        <div style="display:flex;gap:8px">
            <button class="btn btn-sm" onclick="downloadTemplate('csv')">CSV template</button>
            <button class="btn btn-sm" onclick="downloadTemplate('json')">JSON template</button>
            <button class="btn btn-sm" onclick="downloadTemplate('jsonl')">JSONL template</button>
        </div>
    </div>`;

    html += `
        <div class="drop-zone" id="dropZone">
            <div class="drop-zone-text"><strong>Click to upload</strong> or drag and drop</div>
            <div class="drop-zone-formats">CSV, JSON, JSONL (max 50MB)</div>
            <input type="file" id="fileInput" accept=".csv,.json,.jsonl" style="display:none">
        </div>`;

    if (state.datasetId) {
        html += `<div class="dataset-info">
            Dataset loaded: <span class="count">${state.rowCount} rows</span>, ${state.columns.length} columns
            <button class="btn btn-sm" onclick="clearDataset()" style="margin-left:8px;color:var(--danger);padding:2px 8px">Remove</button>
        </div>`;
        html += renderPreviewTable();
    }

    return html;
}

function initStep2() {
    const dz = document.getElementById('dropZone');
    const fi = document.getElementById('fileInput');
    if (!dz || !fi) return;
    dz.addEventListener('click', () => fi.click());
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
    dz.addEventListener('drop', e => {
        e.preventDefault(); dz.classList.remove('dragover');
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
    fi.addEventListener('change', () => { if (fi.files.length) uploadFile(fi.files[0]); });
}

function getRequiredColumns() {
    const cols = [];
    const cm = state.columnMapping;
    const tplVars = getTemplateVars();
    if (cm.input_column) cols.push({ name: cm.input_column, required: true });
    if (cm.expected_output_column) cols.push({ name: cm.expected_output_column, required: false });
    if (cm.context_column) cols.push({ name: cm.context_column, required: false });
    if (cm.tools_called_column) cols.push({ name: cm.tools_called_column, required: false });
    if (cm.expected_tools_column) cols.push({ name: cm.expected_tools_column, required: false });
    tplVars.forEach(v => {
        const mapped = cm.template_variable_map[v] || v;
        if (!cols.find(c => c.name === mapped)) cols.push({ name: mapped, required: true });
    });
    if (cols.length === 0) cols.push({ name: 'input', required: true });
    return cols;
}

function getTemplateVars() {
    const sources = [
        state.apiConfig.body_template,
        state.apiConfig.base_url,
        ...Object.values(state.apiConfig.query_params),
        ...state.apiConfig.headers.map(h => h.value),
    ];
    const all = sources.join(' ');
    return [...new Set([...(all.matchAll(/\{\{(\w+)\}\}/g))].map(m => m[1]))];
}

function downloadTemplate(format) {
    const cols = getRequiredColumns();
    const colNames = cols.map(c => c.name);
    const sampleRow = {};
    colNames.forEach(c => sampleRow[c] = `example_${c}`);

    let content, mime, ext;
    if (format === 'csv') {
        const header = colNames.join(',');
        const row = colNames.map(c => `"example_${c}"`).join(',');
        content = header + '\n' + row + '\n' + colNames.map(c => `"example_${c}_2"`).join(',') + '\n';
        mime = 'text/csv'; ext = 'csv';
    } else if (format === 'json') {
        const rows = [sampleRow, Object.fromEntries(colNames.map(c => [c, `example_${c}_2`]))];
        content = JSON.stringify(rows, null, 2);
        mime = 'application/json'; ext = 'json';
    } else {
        const row2 = Object.fromEntries(colNames.map(c => [c, `example_${c}_2`]));
        content = JSON.stringify(sampleRow) + '\n' + JSON.stringify(row2) + '\n';
        mime = 'application/jsonl'; ext = 'jsonl';
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `dataset_template.${ext}`; a.click();
    URL.revokeObjectURL(url);
}

function clearDataset() {
    if (state.datasetId) fetch(`/api/connector/dataset/${state.datasetId}`, { method: 'DELETE' });
    state.datasetId = null; state.columns = []; state.previewRows = []; state.rowCount = 0;
    renderStep(2);
}

async function uploadFile(file) {
    const form = new FormData();
    form.append('file', file);
    try {
        const resp = await fetch('/api/connector/upload-dataset', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) { alert(data.error || 'Upload failed'); return; }
        state.datasetId = data.dataset_id;
        state.columns = data.columns;
        state.previewRows = data.preview;
        state.rowCount = data.row_count;
        renderStep(2);
    } catch (e) { alert('Upload error: ' + e.message); }
}

function renderPreviewTable() {
    if (!state.previewRows.length) return '';
    let html = '<div class="data-table-wrapper"><table class="data-table"><thead><tr>';
    state.columns.forEach(c => html += `<th>${esc(c)}</th>`);
    html += '</tr></thead><tbody>';
    state.previewRows.forEach(row => {
        html += '<tr>';
        state.columns.forEach(c => {
            const v = row[c];
            html += `<td title="${esc(String(v || ''))}">${esc(v == null ? '' : String(v).substring(0, 100))}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    return html;
}

// ==== STEP 3: Test & Preview ====

function renderStep3() {
    let html = renderStepHeader('Test & Preview', 'Verify your full pipeline: API call + response extraction', 'Next: Execute');

    const tplVars = getTemplateVars();
    if (tplVars.length && state.previewRows.length) {
        html += `<div class="section-card">
            <div class="section-card-title">Sample Request Data (row 1)</div>
            <div style="font-size:0.85em;color:var(--text-secondary);line-height:1.8">`;
        tplVars.forEach(v => {
            const col = state.columnMapping.template_variable_map[v] || v;
            const val = state.previewRows[0]?.[col] || '';
            html += `<div><code style="color:var(--accent)">{{${esc(v)}}}</code> = ${esc(String(val).substring(0, 100))}</div>`;
        });
        html += '</div></div>';
    }

    if (state.testResponse?.response_body) {
        const data = state.testResponse;
        const isOk = data.status_code >= 200 && data.status_code < 400;
        html += `<div class="section-card">
            <div class="section-card-title">Response</div>
            <div class="response-status">
                <span class="status-code ${isOk ? 'success' : 'error'}">${data.status_code}</span>
                <span style="color:var(--text-muted)">${data.elapsed_ms}ms</span>
            </div>
            <div class="json-tree">${renderJsonTree(data.response_body, '', 0)}</div>
        </div>`;

        const mappings = [
            { label: 'actual_output', path: state.responseMapping.actual_output_path },
            { label: 'retrieval_context', path: state.responseMapping.retrieval_context_path },
            { label: 'tools_called', path: state.responseMapping.tools_called_path },
            { label: 'token_usage', path: state.responseMapping.token_usage_path },
            { label: 'system_prompt', path: state.responseMapping.system_prompt_path },
        ].filter(m => m.path);

        if (mappings.length) {
            html += `<div class="section-card"><div class="section-card-title">Extracted Values</div>`;
            mappings.forEach(m => {
                html += `<div style="margin-bottom:8px"><strong style="font-size:0.8em;color:var(--text-secondary)">${m.label}</strong>`;
                try {
                    const extracted = extractPath(data.response_body, m.path);
                    const text = (typeof extracted === 'object' && extracted !== null) ? JSON.stringify(extracted, null, 2) : String(extracted);
                    html += `<div class="mapping-preview">${esc(text)}</div>`;
                } catch {
                    html += `<div style="color:var(--danger);font-size:0.8em">Invalid path: ${esc(m.path)}</div>`;
                }
                html += '</div>';
            });
            html += '</div>';
        }
    } else if (state.testResponse?.error) {
        html += `<div class="error-item">${esc(state.testResponse.error)}</div>`;
    }

    html += `<div id="testResult"></div>
    <div style="margin-top:16px">
        <button class="btn btn-secondary" onclick="testConnectionStep3()">Send Test Request</button>
    </div>`;

    return html;
}

async function testConnectionStep3() {
    saveCurrentStep();
    const a = state.apiConfig;
    if (!a.base_url) { alert('Please set a URL in Settings'); return; }

    const sampleRow = buildTestSampleRow();
    const varMap = {};
    getTemplateVars().forEach(v => {
        varMap[v] = state.columnMapping.template_variable_map[v] || v;
    });

    const body = {
        base_url: a.base_url,
        method: a.method,
        headers: a.headers.filter(h => h.key && h.enabled),
        query_params: a.query_params,
        body_template: a.body_template,
        timeout_seconds: a.timeout_seconds,
        sample_row: sampleRow,
        variable_map: varMap,
    };

    const target = document.getElementById('testResult');
    if (target) target.innerHTML = '<p style="color:var(--text-muted);padding:12px">Sending request...</p>';

    try {
        const resp = await fetch('/api/connector/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        state.testResponse = await resp.json();
        renderStep(3);
    } catch (e) {
        if (target) target.innerHTML = `<div class="error-item">Error: ${esc(e.message)}</div>`;
    }
}

// ==== STEP 4: Execute ====

function renderStep4() {
    const enabledMetrics = Object.entries(state.selectedMetrics).filter(([_, v]) => v.enabled).map(([k]) => k);
    const issues = [];
    if (!state.apiConfig.base_url) issues.push('No API URL configured');
    if (!state.datasetId) issues.push('No dataset uploaded');
    if (!state.responseMapping.actual_output_path) issues.push('No response mapping for actual_output');
    if (!enabledMetrics.length) issues.push('No metrics selected');

    let html = renderStepHeader('Execute Evaluation', 'Review and run', null);

    html += `<div class="section-card">
        <div style="font-size:0.85em;line-height:2;color:var(--text-secondary)">
            <div><strong style="color:var(--text)">URL:</strong> ${esc(state.apiConfig.base_url)} <span style="opacity:0.5">(${state.apiConfig.method})</span></div>
            <div><strong style="color:var(--text)">Dataset:</strong> ${state.rowCount} rows</div>
            <div><strong style="color:var(--text)">Input column:</strong> ${esc(state.columnMapping.input_column)}</div>
            <div><strong style="color:var(--text)">Response path:</strong> ${esc(state.responseMapping.actual_output_path)}</div>
            <div><strong style="color:var(--text)">Eval model:</strong> ${esc(state.evalModel)}</div>
            <div><strong style="color:var(--text)">Metrics (${enabledMetrics.length}):</strong> ${enabledMetrics.join(', ') || 'None'}</div>
        </div>
    </div>`;

    if (issues.length) {
        html += `<div class="error-log" style="margin-bottom:16px">`;
        issues.forEach(i => html += `<div class="error-item">${esc(i)}</div>`);
        html += '</div>';
    }

    html += `<div id="jobArea">
        <button class="btn btn-primary" onclick="startJob()" id="startBtn" ${issues.length ? 'disabled' : ''}>Run Evaluation</button>
    </div>`;

    return html;
}

async function startJob() {
    const enabledMetrics = Object.entries(state.selectedMetrics)
        .filter(([_, v]) => v.enabled)
        .map(([name, v]) => ({ metric_class: name, params: v.params }));

    const jobName = `connector_${Date.now()}`;
    const config = {
        name: jobName,
        api_config: state.apiConfig,
        response_mapping: state.responseMapping,
        dataset_column_mapping: state.columnMapping,
        metrics: enabledMetrics,
        eval_model: state.evalModel,
        cost_per_1m_tokens: state.costPer1mTokens,
    };

    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.disabled = true;

    try {
        const resp = await fetch('/api/connector/start-job', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: state.datasetId, config }),
        });
        const data = await resp.json();
        if (!resp.ok) { alert(data.error || 'Failed'); if (startBtn) startBtn.disabled = false; return; }
        state.jobId = data.job_id;
        pollJobProgress();
    } catch (e) { alert('Error: ' + e.message); if (startBtn) startBtn.disabled = false; }
}

function pollJobProgress() {
    if (state.jobPollingInterval) clearInterval(state.jobPollingInterval);
    const jobArea = document.getElementById('jobArea');
    jobArea.innerHTML = `<div class="progress-container">
        <div class="progress-phase" id="progressPhase">Starting...</div>
        <div class="progress-bar-wrapper"><div class="progress-bar-fill" id="progressBar" style="width:0%"></div></div>
        <div class="progress-stats"><span id="progressCount">0 / 0</span><span id="progressPercent">0%</span><span id="progressTime"></span></div>
        <button class="btn btn-sm btn-secondary" onclick="cancelJob()" id="cancelBtn" style="margin-top:10px">Cancel</button>
        <div class="error-log" id="errorLog"></div>
    </div>`;

    state.jobPollingInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/connector/job/${state.jobId}/progress`);
            const data = await resp.json();
            updateProgress(data);
            if (['completed', 'failed', 'cancelled'].includes(data.status)) {
                clearInterval(state.jobPollingInterval);
                state.jobPollingInterval = null;
            }
        } catch (e) { console.error('Poll error:', e); }
    }, 1000);
}

function updateProgress(data) {
    const phaseEl = document.getElementById('progressPhase');
    const barEl = document.getElementById('progressBar');
    const countEl = document.getElementById('progressCount');
    const percentEl = document.getElementById('progressPercent');
    const errorEl = document.getElementById('errorLog');
    const cancelBtn = document.getElementById('cancelBtn');

    const labels = { api_calls: 'Sending API requests...', evaluation: 'Running evaluation...', done: 'Done' };
    if (phaseEl) phaseEl.textContent = labels[data.current_phase] || data.current_phase;

    const pct = data.total_rows > 0 ? Math.round(data.completed_rows / data.total_rows * 100) : 0;
    if (barEl) barEl.style.width = pct + '%';
    if (countEl) countEl.textContent = `${data.completed_rows} / ${data.total_rows}`;
    if (percentEl) percentEl.textContent = pct + '%';
    const timeEl = document.getElementById('progressTime');
    if (timeEl && data.avg_response_time_ms > 0) timeEl.textContent = `avg ${data.avg_response_time_ms}ms`;
    if (errorEl && data.errors.length) errorEl.innerHTML = data.errors.map(e => `<div class="error-item">${esc(e)}</div>`).join('');

    if (data.status === 'completed') {
        if (phaseEl) phaseEl.textContent = 'Evaluation completed!';
        if (barEl) barEl.style.width = '100%';
        if (cancelBtn) cancelBtn.style.display = 'none';
        document.getElementById('jobArea').innerHTML += `
            <div style="margin-top:16px">
                <a href="/" class="btn btn-primary" style="text-decoration:none;display:inline-block">View Results in Dashboard</a>
            </div>`;
    } else if (data.status === 'failed') {
        if (phaseEl) { phaseEl.textContent = 'Failed'; phaseEl.style.color = 'var(--danger)'; }
        if (cancelBtn) cancelBtn.style.display = 'none';
    } else if (data.status === 'cancelled') {
        if (phaseEl) phaseEl.textContent = 'Cancelled';
        if (cancelBtn) cancelBtn.style.display = 'none';
    }
}

async function cancelJob() {
    if (state.jobId) await fetch(`/api/connector/job/${state.jobId}/cancel`, { method: 'POST' });
}

// ---- Metrics save ----

function saveMetricParams() {
    document.querySelectorAll('[data-metric][data-param]').forEach(el => {
        const metric = el.dataset.metric;
        const param = el.dataset.param;
        if (!state.selectedMetrics[metric]) state.selectedMetrics[metric] = { enabled: false, params: {} };
        let val = el.value;
        const info = state.metricsInfo.find(m => m.name === metric);
        const pinfo = info?.params.find(p => p.name === param);
        if (el.type === 'number') val = val === '' ? null : parseFloat(val);
        else if (pinfo?.type === 'bool' && (val === 'true' || val === 'false')) val = val === 'true';
        if (pinfo?.type === 'list' && typeof val === 'string' && val) val = val.split(',').map(s => s.trim()).filter(s => s);
        state.selectedMetrics[metric].params[param] = val;
    });
}

// ---- Config Save/Load ----

async function loadMetrics() {
    try { state.metricsInfo = await (await fetch('/api/connector/metrics')).json(); } catch {}
}

async function loadConfigs() {
    try { state.savedConfigs = await (await fetch('/api/connector/configs')).json(); } catch {}
}

async function loadProviders() {
    try { state.providers = await (await fetch('/api/connector/providers')).json(); } catch {}
    try {
        const cfg = await (await fetch('/api/connector/custom-llm-config')).json();
        if (cfg) state.customLlmConfig = { base_url: cfg.base_url || '', api_key: cfg.api_key ? '••••••••' : '', model_name: cfg.model_name || '' };
    } catch {}
}

async function saveApiKey(envVar, value) {
    try {
        await fetch('/api/connector/save-api-key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ env_var: envVar, value }),
        });
        await loadProviders();
        renderStep(1);
        showToast('API key saved');
    } catch (e) { alert('Error saving key: ' + e.message); }
}

async function deleteApiKey(envVar) {
    try {
        await fetch('/api/connector/delete-api-key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ env_var: envVar }),
        });
        await loadProviders();
        renderStep(1);
        showToast('API key removed');
    } catch (e) { alert('Error: ' + e.message); }
}

async function saveCustomLlmConfig() {
    const baseUrl = document.getElementById('customLlmBaseUrl')?.value || '';
    const apiKey = document.getElementById('customLlmApiKey')?.value || '';
    const modelName = document.getElementById('customLlmModelName')?.value || '';

    if (!baseUrl) { alert('Base URL is required'); return; }
    if (!modelName) { alert('Model name is required'); return; }

    try {
        const body = { base_url: baseUrl, model_name: modelName };
        if (apiKey && !apiKey.startsWith('••')) body.api_key = apiKey;
        await fetch('/api/connector/custom-llm-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        await loadProviders();
        renderStep(1);
        showToast('Custom LLM configured');
    } catch (e) { alert('Error: ' + e.message); }
}

async function saveProjectConfig() {
    saveCurrentStep();
    const active = state.savedConfigs.find(c => c.id === state.activeConfigId);
    const name = active ? active.name : prompt('Project name:', 'My Project');
    if (!name) return;

    const config = {
        id: state.activeConfigId || undefined,
        name,
        api_config: state.apiConfig,
        response_mapping: state.responseMapping,
        dataset_column_mapping: state.columnMapping,
        metrics: Object.entries(state.selectedMetrics).filter(([_, v]) => v.enabled).map(([n, v]) => ({ metric_class: n, params: v.params })),
        eval_model: state.evalModel,
        cost_per_1m_tokens: state.costPer1mTokens,
    };

    try {
        const resp = await fetch('/api/connector/save-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await resp.json();
        state.activeConfigId = data.config_id;
        await loadConfigs();
        showToast('Project saved');
        renderStep(state.currentStep);
    } catch (e) { alert('Save error: ' + e.message); }
}

async function saveProjectConfigAs() {
    saveCurrentStep();
    const name = prompt('New project name:', 'My Project');
    if (!name) return;

    const config = {
        name,
        api_config: state.apiConfig,
        response_mapping: state.responseMapping,
        dataset_column_mapping: state.columnMapping,
        metrics: Object.entries(state.selectedMetrics).filter(([_, v]) => v.enabled).map(([n, v]) => ({ metric_class: n, params: v.params })),
        eval_model: state.evalModel,
        cost_per_1m_tokens: state.costPer1mTokens,
    };

    try {
        const resp = await fetch('/api/connector/save-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await resp.json();
        state.activeConfigId = data.config_id;
        await loadConfigs();
        showToast('Saved as new project');
        renderStep(state.currentStep);
    } catch (e) { alert('Save error: ' + e.message); }
}

function resetProject() {
    state.activeConfigId = null;
    state.apiConfig = {
        base_url: '', method: 'POST',
        headers: [{ key: 'Content-Type', value: 'application/json', enabled: true }],
        query_params: {}, body_template: '',
        timeout_seconds: 60, max_retries: 1, delay_between_requests_ms: 0,
    };
    state.testResponse = null;
    state.responseMapping = { actual_output_path: '', retrieval_context_path: '', tools_called_path: '', token_usage_path: '', system_prompt_path: '' };
    state.costPer1mTokens = 0;
    state.columnMapping = { input_column: 'input', expected_output_column: '', context_column: '', tools_called_column: '', expected_tools_column: '', template_variable_map: {} };
    state.selectedMetrics = {};
    state.evalModel = 'gpt-4o-mini';
    showToast('New project');
    renderStep(state.currentStep);
}

async function loadSelectedConfig() {
    const val = getSelectValue('configSelect');
    if (!val) return;
    try {
        const data = await (await fetch(`/api/connector/config/${val}`)).json();
        if (data.api_config) {
            state.apiConfig = { ...state.apiConfig, ...data.api_config,
                headers: (data.api_config.headers || []).map(h => typeof h === 'object' ? h : { key: '', value: '', enabled: true }) };
        }
        if (data.response_mapping) state.responseMapping = { ...state.responseMapping, ...data.response_mapping };
        if (data.dataset_column_mapping) state.columnMapping = { ...state.columnMapping, ...data.dataset_column_mapping };
        if (data.eval_model) state.evalModel = data.eval_model;
        if (data.cost_per_1m_tokens != null) state.costPer1mTokens = data.cost_per_1m_tokens;
        if (data.metrics) {
            state.selectedMetrics = {};
            data.metrics.forEach(m => { state.selectedMetrics[m.metric_class] = { enabled: true, params: m.params || {} }; });
        }
        state.activeConfigId = val;
        state.testResponse = null;
        showToast('Project loaded');
        renderStep(state.currentStep);
    } catch (e) { alert('Load error: ' + e.message); }
}

async function deleteCurrentConfig() {
    if (!state.activeConfigId) return;
    const active = state.savedConfigs.find(c => c.id === state.activeConfigId);
    if (!confirm(`Delete project "${active?.name}"?`)) return;
    await fetch(`/api/connector/config/${state.activeConfigId}`, { method: 'DELETE' });
    state.activeConfigId = null;
    await loadConfigs();
    showToast('Project deleted');
    renderStep(state.currentStep);
}

// ---- JSON Tree ----

let _jsonNodeId = 0;

function renderJsonTree(obj, path, depth) {
    depth = depth || 0;
    if (depth > 10) return '<span class="json-null">...</span>';
    if (obj === null) return '<span class="json-null">null</span>';
    if (typeof obj === 'boolean') return `<span class="json-bool">${obj}</span>`;
    if (typeof obj === 'number') return `<span class="json-number json-clickable" onclick="copyPath('${esc(path)}')" title="${esc(path)}">${obj}</span>`;
    if (typeof obj === 'string') {
        return `<span class="json-string json-clickable" onclick="copyPath('${esc(path)}')" title="${esc(path)}">"${esc(obj)}"</span>`;
    }
    const indent = '  '.repeat(depth), inner = '  '.repeat(depth + 1);
    if (Array.isArray(obj)) {
        if (!obj.length) return '[]';
        const id = 'jn' + (++_jsonNodeId);
        const collapsed = depth >= 2;
        const preview = `<span class="json-muted">Array(${obj.length})</span>`;
        let h = `<span class="json-toggle ${collapsed ? 'collapsed' : ''}" onclick="toggleJsonNode('${id}')">${collapsed ? '\u25b6' : '\u25bc'}</span>[`;
        h += `<span class="json-collapse-preview" id="${id}_p" style="display:${collapsed ? 'inline' : 'none'}">${preview}]</span>`;
        h += `<span id="${id}" style="display:${collapsed ? 'none' : 'inline'}">\n`;
        obj.forEach((item, i) => { h += inner + renderJsonTree(item, path ? `${path}[${i}]` : `[${i}]`, depth + 1) + (i < obj.length - 1 ? ',' : '') + '\n'; });
        h += indent + ']</span>';
        return h;
    }
    if (typeof obj === 'object') {
        const keys = Object.keys(obj);
        if (!keys.length) return '{}';
        const id = 'jn' + (++_jsonNodeId);
        const collapsed = depth >= 3;
        const previewKeys = keys.slice(0, 3).join(', ') + (keys.length > 3 ? ', ...' : '');
        const preview = `<span class="json-muted">{${previewKeys}}</span>`;
        let h = `<span class="json-toggle ${collapsed ? 'collapsed' : ''}" onclick="toggleJsonNode('${id}')">${collapsed ? '\u25b6' : '\u25bc'}</span>{`;
        h += `<span class="json-collapse-preview" id="${id}_p" style="display:${collapsed ? 'inline' : 'none'}">${preview}}</span>`;
        h += `<span id="${id}" style="display:${collapsed ? 'none' : 'inline'}">\n`;
        keys.forEach((key, i) => { h += inner + `<span class="json-key">"${esc(key)}"</span>: ` + renderJsonTree(obj[key], path ? `${path}.${key}` : key, depth + 1) + (i < keys.length - 1 ? ',' : '') + '\n'; });
        h += indent + '}</span>';
        return h;
    }
    return String(obj);
}

function toggleJsonNode(id) {
    const body = document.getElementById(id);
    const preview = document.getElementById(id + '_p');
    const toggle = body?.previousElementSibling?.previousElementSibling;
    if (!body) return;
    const isHidden = body.style.display === 'none';
    body.style.display = isHidden ? 'inline' : 'none';
    if (preview) preview.style.display = isHidden ? 'none' : 'inline';
    if (toggle) { toggle.textContent = isHidden ? '\u25bc' : '\u25b6'; toggle.classList.toggle('collapsed', !isHidden); }
}

function copyPath(path) {
    navigator.clipboard?.writeText(path);
    const el = document.getElementById('mapActualOutput');
    if (el && !el.value) {
        el.value = path;
        state.responseMapping.actual_output_path = path;
    }
    showToast(`Copied: ${path}`);
    updateMappingPreview();
}

function updateMappingPreview() {
    if (!state.testResponse?.response_body) return;
    const data = state.testResponse.response_body;
    const previewFields = [
        { inputId: 'mapActualOutput', previewId: 'preview_actual_output' },
        { inputId: 'mapRetrievalContext', previewId: 'preview_retrieval_context' },
        { inputId: 'mapToolsCalled', previewId: 'preview_tools_called' },
        { inputId: 'mapTokenUsage', previewId: 'preview_token_usage' },
        { inputId: 'mapSystemPrompt', previewId: 'preview_system_prompt' },
    ];
    previewFields.forEach(f => {
        const input = document.getElementById(f.inputId);
        const preview = document.getElementById(f.previewId);
        if (!input || !preview) return;
        const path = input.value.trim();
        if (!path) { preview.innerHTML = ''; preview.className = 'mapping-live-preview'; return; }
        const extracted = extractPath(data, path);
        if (extracted === undefined || extracted === null) {
            preview.innerHTML = `<span class="preview-label">No match for this path</span>`;
            preview.className = 'mapping-live-preview invalid';
        } else {
            const text = typeof extracted === 'object' ? JSON.stringify(extracted, null, 2) : String(extracted);
            preview.innerHTML = `<span class="preview-label">Extracted:</span>\n<span class="preview-value">${esc(text)}</span>`;
            preview.className = 'mapping-live-preview valid';
        }
    });
}

function showToast(msg, type) {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        toast.className = 'toast';
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.remove('toast-error');
    if (type === 'error') toast.classList.add('toast-error');
    toast.classList.add('visible');
    setTimeout(() => toast.classList.remove('visible'), 2500);
}

function extractPath(obj, path) {
    if (!path) return obj;
    if (path.startsWith('$.')) path = path.substring(2);
    else if (path === '$') return obj;
    if (!path) return obj;
    return _extractRecursive(obj, path);
}

function _extractRecursive(data, pathStr) {
    if (data == null || !pathStr) return data;
    let result = data;
    const segments = pathStr.match(/([^.\[\]]+|\[[^\]]*\])/g) || [];

    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        if (result == null) return undefined;

        if (seg.startsWith('[') && seg.endsWith(']')) {
            const inner = seg.slice(1, -1);

            // Wildcard [*]
            if (inner === '*') {
                if (!Array.isArray(result)) return undefined;
                const rest = segments.slice(i + 1);
                if (rest.length > 0) {
                    const restPath = rest.map(s => s.startsWith('[') ? s : '.' + s).join('').replace(/^\./, '');
                    return result.map(item => _extractRecursive(item, restPath)).filter(v => v != null);
                }
                return result;
            }

            // Filter [?(@.field==value)]
            if (inner.startsWith('?')) {
                if (!Array.isArray(result)) return undefined;
                const fm = inner.match(/\?\(@\.([^=!<>]+)\s*([=!<>]+)\s*['"]([^'"]*)['"]\)/);
                if (fm) {
                    const [, fieldPath, op, val] = fm;
                    const filtered = result.filter(item => {
                        if (!item) return false;
                        let fv = item;
                        for (const p of fieldPath.split('.')) fv = fv?.[p];
                        if (op === '==' || op === '=') return fv == val;
                        if (op === '!=') return fv != val;
                        return false;
                    });
                    const rest = segments.slice(i + 1);
                    if (rest.length > 0) {
                        const restPath = rest.map(s => s.startsWith('[') ? s : '.' + s).join('').replace(/^\./, '');
                        return filtered.map(item => _extractRecursive(item, restPath)).filter(v => v != null);
                    }
                    return filtered;
                }
                return undefined;
            }

            // Slice [-1:]
            if (inner.endsWith(':')) {
                const si = parseInt(inner.slice(0, -1));
                if (isNaN(si) || !Array.isArray(result)) return undefined;
                const ai = si < 0 ? result.length + si : si;
                result = (ai >= 0 && ai < result.length) ? result[ai] : undefined;
                continue;
            }

            // Numeric index (positive or negative)
            const idx = parseInt(inner);
            if (isNaN(idx)) return undefined;
            if (!Array.isArray(result)) return undefined;
            const ai = idx < 0 ? result.length + idx : idx;
            result = (ai >= 0 && ai < result.length) ? result[ai] : undefined;
        } else {
            if (typeof result !== 'object' || result === null || !(seg in result)) return undefined;
            result = result[seg];
        }
    }
    return result;
}

// ---- Utils ----
function esc(str) {
    if (str == null) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}
