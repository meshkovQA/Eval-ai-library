HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval AI Library - Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="{{ url_for('static', filename='dashboard.css') }}">
</head>
<body>
    <nav class="top-nav">
        <div class="nav-brand">Eval AI</div>
        <div class="nav-links">
            <a href="/" class="nav-link active">Dashboard</a>
            <a href="/connector" class="nav-link">API Connector</a>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme" id="themeBtn">&#9790;</button>
        </div>
    </nav>
    <script>
    (function(){var t=localStorage.getItem('eval-theme');if(t)document.documentElement.setAttribute('data-theme',t);
    var btn=document.getElementById('themeBtn');if(!btn)return;
    var isDark=t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme:dark)').matches);
    btn.textContent=isDark?'\u2600':'\u263E';})();
    function toggleTheme(){var r=document.documentElement,c=r.getAttribute('data-theme');
    var isDark=c==='dark'||(c!=='light'&&window.matchMedia('(prefers-color-scheme:dark)').matches);
    var next=isDark?'light':'dark';r.setAttribute('data-theme',next);localStorage.setItem('eval-theme',next);
    document.getElementById('themeBtn').textContent=next==='dark'?'\u2600':'\u263E';}
    </script>

    <div class="container">
        <header>
            <div>
                <h1>Eval AI Library Dashboard</h1>
                <div class="timestamp" id="timestamp">Loading...</div>
            </div>
            <div class="controls">
                <select id="sessionSelect" onchange="loadSession()">
                    <option value="">Loading sessions...</option>
                </select>
                <button onclick="refreshData()">Refresh</button>
                <button class="primary" onclick="clearCache()">Clear Cache</button>
            </div>
        </header>
        
        <div id="content" class="loading">
            Loading data...
        </div>
    </div>
    
    <!-- Modal for detailed information -->
    <div id="detailsModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="test-status">
                    <h2 id="modalTitle">Test Details</h2>
                </div>
                <span class="close" onclick="closeModal()">&times;</span>
            </div>
            <div class="modal-body" id="modalBody"></div>
        </div>
    </div>
    
<script src="{{ url_for('static', filename='dashboard.js') }}"></script>
</body>
</html>
"""
