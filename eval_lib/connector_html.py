CONNECTOR_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval AI Library - API Connector</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='dashboard.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='connector.css') }}">
</head>
<body>
    <nav class="top-nav">
        <div class="nav-brand">Eval AI</div>
        <div class="nav-links">
            <a href="/" class="nav-link">Dashboard</a>
            <a href="/connector" class="nav-link active">API Connector</a>
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

    <div class="connector-container">
        <div class="connector-layout">
            <!-- Stepper sidebar -->
            <div class="stepper">
                <div class="step active" data-step="1">
                    <div class="step-number">1</div>
                    <div class="step-label">Settings</div>
                </div>
                <div class="step" data-step="2">
                    <div class="step-number">2</div>
                    <div class="step-label">Dataset</div>
                </div>
                <div class="step" data-step="3">
                    <div class="step-number">3</div>
                    <div class="step-label">Test</div>
                </div>
                <div class="step" data-step="4">
                    <div class="step-number">4</div>
                    <div class="step-label">Execute</div>
                </div>
            </div>

            <!-- Main content -->
            <div class="step-content" id="stepContent">
                <!-- Rendered by JS -->
            </div>
        </div>
    </div>

<script src="{{ url_for('static', filename='connector.js') }}"></script>
</body>
</html>
"""
