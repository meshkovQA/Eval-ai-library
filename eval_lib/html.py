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
