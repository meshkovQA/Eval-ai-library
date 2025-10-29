HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval AI Library - Interactive Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }
        
        h1 {
            color: #667eea;
            font-size: 2.5em;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        select, button {
            padding: 10px 20px;
            border-radius: 8px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        button:hover {
            background: #667eea;
            color: white;
        }
        
        .timestamp {
            color: #666;
            font-size: 0.9em;
            margin-left: 20px;
        }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        
        .summary-card:hover {
            transform: translateY(-5px);
        }
        
        .summary-card h3 {
            font-size: 0.9em;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .metric-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .metric-score {
            font-size: 2.5em;
            font-weight: bold;
            color: #764ba2;
            margin-bottom: 15px;
        }
        
        .metric-details p {
            margin: 8px 0;
            color: #555;
            font-size: 0.9em;
        }
        
        .charts {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .chart-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        .chart-container h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }
        
        th:hover {
            background: linear-gradient(135deg, #5568d3 0%, #653a8b 100%);
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
        }
        
        tr.success {
            background: #f0fdf4;
        }
        
        tr.failed {
            background: #fef2f2;
        }
        
        tr:hover {
            background: #f8f9fa !important;
        }
        
        .reason {
            max-width: 300px;
            color: #666;
        }
        
        .view-details-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 5px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.3s;
        }
        
        .view-details-btn:hover {
            background: #5568d3;
            transform: scale(1.05);
        }
        
        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.7);
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .modal-content {
            background-color: #fefefe;
            margin: 2% auto;
            padding: 30px;
            border-radius: 15px;
            width: 90%;
            max-width: 900px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: slideIn 0.3s;
        }
        
        @keyframes slideIn {
            from {
                transform: translateY(-50px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #667eea;
        }
        
        .modal-header h2 {
            color: #667eea;
            margin: 0;
        }
        
        .close {
            color: #aaa;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }
        
        .close:hover {
            color: #667eea;
        }
        
        .detail-section {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .detail-section h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .detail-section pre {
            background: white;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85em;
            line-height: 1.5;
        }
        
        .detail-section p {
            margin: 8px 0;
            color: #555;
            line-height: 1.6;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            margin-right: 8px;
        }
        
        .badge-success {
            background: #d1fae5;
            color: #065f46;
        }
        
        .badge-failed {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
            font-size: 1.2em;
        }
        
        .no-data {
            text-align: center;
            padding: 60px;
            color: #999;
        }
        
        .no-data h2 {
            color: #667eea;
            margin-bottom: 20px;
        }
        
        @media (max-width: 768px) {
            .charts {
                grid-template-columns: 1fr;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            header {
                flex-direction: column;
                gap: 15px;
            }
            
            .modal-content {
                width: 95%;
                margin: 5% auto;
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>📊 Eval AI Library Dashboard</h1>
                <span class="timestamp" id="timestamp">Loading...</span>
            </div>
            <div class="controls">
                <select id="sessionSelect" onchange="loadSession()">
                    <option value="">Loading sessions...</option>
                </select>
                <button onclick="refreshData()">🔄 Refresh</button>
                <button onclick="clearCache()">🗑️ Clear Cache</button>
            </div>
        </header>
        
        <div id="content" class="loading">
            Loading data...
        </div>
    </div>
    
    <!-- Modal для детальной информации -->
    <div id="detailsModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>📋 Evaluation Details</h2>
                <span class="close" onclick="closeModal()">&times;</span>
            </div>
            <div id="modalBody"></div>
        </div>
    </div>
    
    <script>
        let currentData = null;
        let scoresChart = null;
        let successChart = null;
        
        // Загрузить список сессий
        async function loadSessions() {
            try {
                const response = await fetch('/api/sessions');
                const sessions = await response.json();
                
                const select = document.getElementById('sessionSelect');
                select.innerHTML = '<option value="latest">Latest Results</option>';
                
                sessions.reverse().forEach(session => {
                    const option = document.createElement('option');
                    option.value = session.session_id;
                    option.textContent = `${session.session_id} (${session.timestamp}) - ${session.total_tests} tests`;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading sessions:', error);
            }
        }
        
        // Загрузить данные сессии
        async function loadSession() {
            const select = document.getElementById('sessionSelect');
            const sessionId = select.value;
            
            try {
                let url = '/api/latest';
                if (sessionId && sessionId !== 'latest') {
                    url = `/api/session/${sessionId}`;
                }
                
                const response = await fetch(url);
                if (!response.ok) {
                    showNoData();
                    return;
                }
                
                const session = await response.json();
                currentData = session.data;
                renderDashboard(session);
            } catch (error) {
                console.error('Error loading session:', error);
                showNoData();
            }
        }
        
        // Показать "нет данных"
        function showNoData() {
            document.getElementById('content').innerHTML = `
                <div class="no-data">
                    <h2>No evaluation results available</h2>
                    <p>Run an evaluation with <code>show_dashboard=True</code> to see results here.</p>
                </div>
            `;
        }
        
        // Отрисовать дашборд
        function renderDashboard(session) {
            const data = session.data;
            document.getElementById('timestamp').textContent = `Generated: ${session.timestamp}`;
            
            const metricsLabels = Object.keys(data.metrics_summary);
            const metricsScores = metricsLabels.map(m => data.metrics_summary[m].avg_score);
            const metricsSuccessRates = metricsLabels.map(m => data.metrics_summary[m].success_rate);
            
            let metricCards = '';
            for (const [metricName, metricData] of Object.entries(data.metrics_summary)) {
                metricCards += `
                    <div class="metric-card">
                        <h3>${metricName}</h3>
                        <div class="metric-score">${metricData.avg_score.toFixed(3)}</div>
                        <div class="metric-details">
                            <p>✅ Passed: ${metricData.passed}</p>
                            <p>❌ Failed: ${metricData.failed}</p>
                            <p>📊 Success Rate: ${metricData.success_rate.toFixed(1)}%</p>
                            <p>🎯 Threshold: ${metricData.threshold}</p>
                            <p>🤖 Model: ${metricData.model}</p>
                            <p>💰 Total Cost: $${metricData.total_cost.toFixed(6)}</p>
                        </div>
                    </div>
                `;
            }
            
            let tableRows = '';
            data.test_cases.forEach((testCase, tcIdx) => {
                testCase.metrics.forEach((metric, mIdx) => {
                    const statusEmoji = metric.success ? '✅' : '❌';
                    const statusClass = metric.success ? 'success' : 'failed';
                    
                    tableRows += `
                        <tr class="${statusClass}">
                            <td>${testCase.test_index}</td>
                            <td>${testCase.input}</td>
                            <td>${metric.name}</td>
                            <td>${metric.score.toFixed(3)}</td>
                            <td>${metric.threshold}</td>
                            <td>${statusEmoji}</td>
                            <td>${metric.evaluation_model}</td>
                            <td>$${(metric.evaluation_cost || 0).toFixed(6)}</td>
                            <td>
                                <button class="view-details-btn" onclick="showDetails(${tcIdx}, ${mIdx})">
                                    View Details
                                </button>
                            </td>
                        </tr>
                    `;
                });
            });
            
            document.getElementById('content').innerHTML = `
                <div class="summary">
                    <div class="summary-card">
                        <h3>Total Tests</h3>
                        <div class="value">${data.total_tests}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Total Cost</h3>
                        <div class="value">$${data.total_cost.toFixed(6)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Metrics</h3>
                        <div class="value">${metricsLabels.length}</div>
                    </div>
                </div>
                
                <h2 style="color: #667eea; margin-bottom: 20px;">📈 Metrics Summary</h2>
                <div class="metrics-grid">
                    ${metricCards}
                </div>
                
                <h2 style="color: #667eea; margin-bottom: 20px;">📊 Charts</h2>
                <div class="charts">
                    <div class="chart-container">
                        <h2>Average Scores by Metric</h2>
                        <canvas id="scoresChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h2>Success Rate by Metric</h2>
                        <canvas id="successChart"></canvas>
                    </div>
                </div>
                
                <h2 style="color: #667eea; margin: 40px 0 20px 0;">📋 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Test #</th>
                            <th>Input</th>
                            <th>Metric</th>
                            <th>Score</th>
                            <th>Threshold</th>
                            <th>Status</th>
                            <th>Model</th>
                            <th>Cost</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tableRows}
                    </tbody>
                </table>
            `;
            
            renderCharts(metricsLabels, metricsScores, metricsSuccessRates);
        }
        
        // Показать детали в модальном окне
        function showDetails(testCaseIdx, metricIdx) {
            const testCase = currentData.test_cases[testCaseIdx];
            const metric = testCase.metrics[metricIdx];
            
            const statusBadge = metric.success 
                ? '<span class="badge badge-success">✅ PASSED</span>'
                : '<span class="badge badge-failed">❌ FAILED</span>';
            
            let modalContent = `
                <div class="detail-section">
                    <h3>Test Case #${testCase.test_index}</h3>
                    <p><strong>Input:</strong> ${testCase.input_full}</p>
                    <p><strong>Actual Output:</strong> ${testCase.actual_output_full || 'N/A'}</p>
                    <p><strong>Expected Output:</strong> ${testCase.expected_output_full || 'N/A'}</p>
                </div>
                
                <div class="detail-section">
                    <h3>Metric: ${metric.name}</h3>
                    ${statusBadge}
                    <p><strong>Score:</strong> ${metric.score.toFixed(3)} / ${metric.threshold}</p>
                    <p><strong>Model:</strong> ${metric.evaluation_model}</p>
                    <p><strong>Cost:</strong> $${(metric.evaluation_cost || 0).toFixed(6)}</p>
                </div>
                
                <div class="detail-section">
                    <h3>Reason</h3>
                    <p>${metric.reason_full || metric.reason}</p>
                </div>
            `;
            
            // Добавляем retrieval context если есть
            if (testCase.retrieval_context && testCase.retrieval_context.length > 0) {
                modalContent += `
                    <div class="detail-section">
                        <h3>Retrieval Context (${testCase.retrieval_context.length} chunks)</h3>
                        ${testCase.retrieval_context.map((ctx, idx) => `
                            <p><strong>Chunk ${idx + 1}:</strong></p>
                            <p style="margin-left: 20px; color: #666;">${ctx.substring(0, 300)}${ctx.length > 300 ? '...' : ''}</p>
                        `).join('')}
                    </div>
                `;
            }
            
            // Добавляем evaluation log если есть
            if (metric.evaluation_log) {
                modalContent += `
                    <div class="detail-section">
                        <h3>Evaluation Log</h3>
                        <pre>${JSON.stringify(metric.evaluation_log, null, 2)}</pre>
                    </div>
                `;
            }
            
            document.getElementById('modalBody').innerHTML = modalContent;
            document.getElementById('detailsModal').style.display = 'block';
        }
        
        // Закрыть модальное окно
        function closeModal() {
            document.getElementById('detailsModal').style.display = 'none';
        }
        
        // Закрытие по клику вне модального окна
        window.onclick = function(event) {
            const modal = document.getElementById('detailsModal');
            if (event.target == modal) {
                closeModal();
            }
        }
        
        // Отрисовать графики
        function renderCharts(labels, scores, successRates) {
            if (scoresChart) scoresChart.destroy();
            if (successChart) successChart.destroy();
            
            const scoresCtx = document.getElementById('scoresChart').getContext('2d');
            scoresChart = new Chart(scoresCtx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Average Score',
                        data: scores,
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0
                        }
                    }
                }
            });
            
            const successCtx = document.getElementById('successChart').getContext('2d');
            successChart = new Chart(successCtx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Success Rate (%)',
                        data: successRates,
                        backgroundColor: [
                            'rgba(102, 126, 234, 0.8)',
                            'rgba(118, 75, 162, 0.8)',
                            'rgba(237, 100, 166, 0.8)',
                            'rgba(255, 154, 158, 0.8)',
                            'rgba(250, 208, 196, 0.8)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true
                }
            });
        }
        
        // Обновить данные
        function refreshData() {
            loadSessions();
            loadSession();
        }
        
        // Очистить кеш
        async function clearCache() {
            if (confirm('Are you sure you want to clear all cached results?')) {
                try {
                    await fetch('/api/clear');
                    alert('Cache cleared!');
                    refreshData();
                } catch (error) {
                    console.error('Error clearing cache:', error);
                    alert('Error clearing cache');
                }
            }
        }
        
        // Инициализация
        loadSessions();
        loadSession();
    </script>
</body>
</html>
"""
