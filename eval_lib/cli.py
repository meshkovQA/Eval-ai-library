# eval_lib/cli.py
"""
Command-line interface for Eval AI Library
"""

import argparse
import sys
from pathlib import Path
import os
import json


def run_dashboard():
    """Run dashboard server from CLI"""
    parser = argparse.ArgumentParser(
        description='Eval AI Library Dashboard Server',
        prog='eval-lib dashboard'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=14500,
        help='Port to run dashboard on (default: 14500)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='.eval_cache',
        help='Path to cache directory (default: .eval_cache)'
    )

    args = parser.parse_args(sys.argv[2:])  # Skip 'eval-lib' and 'dashboard'

    # Import here to avoid loading everything for --help
    from eval_lib.dashboard_server import DashboardCache
    from eval_lib.html import HTML_TEMPLATE
    from flask import Flask, render_template_string, jsonify

    # Create cache with custom directory
    def get_fresh_cache():
        """Reload cache from disk"""
        return DashboardCache(cache_dir=args.cache_dir)

    cache = get_fresh_cache()

    print("="*70)
    print("📊 Eval AI Library - Dashboard Server")
    print("="*70)

    # Check cache
    latest = cache.get_latest()
    if latest:
        print(f"\n✅ Found cached results:")
        print(f"   Latest session: {latest['session_id']}")
        print(f"   Timestamp: {latest['timestamp']}")
        print(f"   Total sessions: {len(cache.get_all())}")
    else:
        print("\n⚠️  No cached results found")
        print("   Run an evaluation with show_dashboard=True to populate cache")

    print(f"\n🚀 Starting server...")
    print(f"   URL: http://localhost:{args.port}")
    print(f"   Host: {args.host}")
    print(f"   Cache: {Path(args.cache_dir).absolute()}")
    print(f"\n💡 Keep this terminal open to keep the server running")
    print(f"   Press Ctrl+C to stop\n")
    print("="*70 + "\n")

    static_folder = os.path.join(os.path.dirname(__file__), 'static')

    app = Flask(__name__, static_folder=static_folder)
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['JSON_SORT_KEYS'] = False

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/favicon.ico')
    def favicon():
        return '', 204

    @app.after_request
    def after_request(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @app.route('/api/latest')
    def api_latest():
        cache = get_fresh_cache()
        latest = cache.get_latest()
        if latest:
            json_str = json.dumps(latest, ensure_ascii=False, sort_keys=False)
            from flask import Response
            return Response(
                json_str,
                mimetype='application/json',
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )
        return jsonify({'error': 'No results available'}), 404

    @app.route('/api/sessions')
    def api_sessions():
        cache = get_fresh_cache()
        sessions = [
            {
                'session_id': s['session_id'],
                'timestamp': s['timestamp'],
                'total_tests': s['data']['total_tests']
            }
            for s in cache.get_all()
        ]
        return jsonify(sessions)

    @app.route('/api/session/<session_id>')
    def api_session(session_id):
        cache = get_fresh_cache()
        session = cache.get_by_session(session_id)
        if session:
            json_str = json.dumps(session, ensure_ascii=False, sort_keys=False)
            from flask import Response
            return Response(
                json_str,
                mimetype='application/json',
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )
        return jsonify({'error': 'Session not found'}), 404

    @app.route('/api/clear')
    def api_clear():
        cache = get_fresh_cache()
        cache.clear()
        return jsonify({'message': 'Cache cleared'})

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard server stopped")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Eval AI Library CLI',
        usage='eval-lib <command> [options]'
    )
    parser.add_argument(
        'command',
        help='Command to run (dashboard, version, help)'
    )

    # Parse only the command
    args = parser.parse_args(sys.argv[1:2])

    if args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'version':
        from eval_lib import __version__
        print(f"Eval AI Library v{__version__}")
    elif args.command == 'help':
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")
        print("Available commands: dashboard, version, help")
        sys.exit(1)


if __name__ == '__main__':
    main()
