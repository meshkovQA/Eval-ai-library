// dashboard.js
let currentData = null;
let scoresChart = null;
let passFailChart = null;
let rawEvaluationLogs = new Map();

// Initialize marked for markdown
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true
    });
}

// Function to render markdown
function renderMarkdown(text) {
    if (!text) return '';

    if (typeof marked !== 'undefined') {
        try {
            return marked.parse(text);
        } catch (error) {
            return text;
        }
    }
    return text;
}

// Toggle collapsible content
function toggleCollapse(id) {
    const element = document.getElementById(id);
    if (!element) return;

    const button = element.nextElementSibling;

    if (element.classList.contains('expanded')) {
        element.classList.remove('expanded');
        if (button && button.classList.contains('see-all')) {
            button.textContent = 'See all';
        }
    } else {
        element.classList.add('expanded');
        if (button && button.classList.contains('see-all')) {
            button.textContent = 'Collapse';
        }
    }
}

// Load sessions list
async function loadSessions() {
    try {
        const response = await fetch('/api/sessions');

        if (!response.ok) {
            return;
        }

        const sessions = await response.json();

        const select = document.getElementById('sessionSelect');
        select.innerHTML = '<option value="latest">Latest Results</option>';

        if (sessions && sessions.length > 0) {
            sessions.reverse().forEach(session => {
                const option = document.createElement('option');
                option.value = session.session_id;
                option.textContent = `${session.session_id} (${session.timestamp}) - ${session.total_tests} tests`;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

// Load session data
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

        const rawText = await response.text();

        // КРИТИЧНО: Извлекаем RAW evaluation_log ДО парсинга
        extractRawEvaluationLogsFromText(rawText);

        // Теперь парсим как обычно
        const session = JSON.parse(rawText);

        if (!session || !session.data) {
            showNoData();
            return;
        }

        currentData = session.data;
        renderDashboard(session);
    } catch (error) {
        console.error('Error loading session:', error);
        showNoData();
    }
}

function extractRawEvaluationLogsFromText(rawText) {
    rawEvaluationLogs.clear();

    try {
        let logIndex = 0;
        let searchPos = 0;

        while (true) {
            // Ищем начало evaluation_log
            const startMatch = rawText.indexOf('"evaluation_log":', searchPos);
            if (startMatch === -1) break;

            // Находим начало JSON объекта после ":"
            let jsonStart = rawText.indexOf('{', startMatch);
            if (jsonStart === -1) break;

            // Находим конец объекта с учетом вложенности
            let depth = 0;
            let pos = jsonStart;
            let inString = false;
            let escape = false;

            while (pos < rawText.length) {
                const char = rawText[pos];

                if (escape) {
                    escape = false;
                    pos++;
                    continue;
                }

                if (char === '\\') {
                    escape = true;
                    pos++;
                    continue;
                }

                if (char === '"') {
                    inString = !inString;
                }

                if (!inString) {
                    if (char === '{') depth++;
                    if (char === '}') depth--;

                    if (depth === 0) {
                        // Нашли конец объекта
                        const rawLogStr = rawText.substring(jsonStart, pos + 1);

                        console.log(`=== Извлечен RAW evaluation_log #${logIndex} ===`);
                        console.log('Длина:', rawLogStr.length, 'символов');
                        console.log('Первые 200 символов:', rawLogStr.substring(0, 200));

                        rawEvaluationLogs.set(logIndex, rawLogStr);
                        logIndex++;

                        searchPos = pos + 1;
                        break;
                    }
                }

                pos++;
            }
        }

        console.log(`=== Всего извлечено ${logIndex} evaluation_log ===`);
    } catch (error) {
        console.error('Error extracting raw evaluation logs:', error);
    }
}

function formatEvaluationLogFromRaw(rawLogStr) {
    if (!rawLogStr || typeof rawLogStr !== 'string') {
        return 'No log available.';
    }

    console.log('=== formatEvaluationLogFromRaw ===');
    console.log('Входящая RAW строка (первые 200 символов):', rawLogStr.substring(0, 200));

    // НОВЫЙ ЛОГ: покажем порядок ключей в RAW строке
    const keyOrder = [];
    const keyRegex = /"([^"]+)"\s*:/g;
    let keyMatch;
    while ((keyMatch = keyRegex.exec(rawLogStr)) !== null) {
        keyOrder.push(keyMatch[1]);
    }
    console.log('=== ПОРЯДОК КЛЮЧЕЙ в RAW строке ===');
    console.log(keyOrder);

    // Декодируем unicode escapes
    const decodedStr = decodeUnicodeEscapes(rawLogStr);
    console.log('После декодирования (первые 200 символов):', decodedStr.substring(0, 200));

    let output = '';

    // Извлекаем пары ключ-значение построчно с сохранением порядка
    // Убираем внешние фигурные скобки
    let content = decodedStr.trim();
    if (content.startsWith('{')) content = content.substring(1);
    if (content.endsWith('}')) content = content.substring(0, content.length - 1);

    // Разбиваем по запятым верхнего уровня
    const pairs = splitTopLevel(content);

    console.log(`Извлечено ${pairs.length} пар ключ-значение`);

    // Создаем map для поиска комментариев
    const pairsMap = new Map();
    pairs.forEach(pair => {
        const colonPos = pair.indexOf(':');
        if (colonPos === -1) return;

        const keyMatch = pair.substring(0, colonPos).trim().match(/"([^"]+)"/);
        if (!keyMatch) return;

        const key = keyMatch[1];
        const value = pair.substring(colonPos + 1).trim();
        pairsMap.set(key, value);
    });

    // Обрабатываем каждую пару
    pairs.forEach(pair => {
        const colonPos = pair.indexOf(':');
        if (colonPos === -1) return;

        const keyMatch = pair.substring(0, colonPos).trim().match(/"([^"]+)"/);
        if (!keyMatch) return;

        const key = keyMatch[1];
        const value = pair.substring(colonPos + 1).trim();

        // Пропускаем comment_ ключи
        if (key.startsWith('comment_')) return;

        // Ищем комментарий
        const commentKey = 'comment_' + key;
        const commentValue = pairsMap.get(commentKey);
        const comment = commentValue ? `  // ${commentValue.replace(/^"|"$/g, '')}` : '';

        // Форматируем значение для вывода
        let prettyValue = formatValue(value);

        output += `<span class="log-key">${escapeHtml(key)}:</span> ${escapeHtml(prettyValue)}${escapeHtml(comment)}\n\n`;
    });

    return output.trim();
}

// Декодирует unicode escapes (\uXXXX)
function decodeUnicodeEscapes(str) {
    return str.replace(/\\u([0-9a-fA-F]{4})/g, (match, hex) => {
        return String.fromCharCode(parseInt(hex, 16));
    });
}

// Разбивает строку по запятым верхнего уровня (не внутри объектов/массивов)
function splitTopLevel(str) {
    const result = [];
    let current = '';
    let depth = 0;
    let inString = false;
    let escape = false;

    for (let i = 0; i < str.length; i++) {
        const char = str[i];

        if (escape) {
            current += char;
            escape = false;
            continue;
        }

        if (char === '\\') {
            current += char;
            escape = true;
            continue;
        }

        if (char === '"') {
            inString = !inString;
            current += char;
            continue;
        }

        if (!inString) {
            if (char === '{' || char === '[') depth++;
            if (char === '}' || char === ']') depth--;

            if (char === ',' && depth === 0) {
                if (current.trim()) result.push(current.trim());
                current = '';
                continue;
            }
        }

        current += char;
    }

    if (current.trim()) result.push(current.trim());
    return result;
}

// Форматирует значение для красивого вывода
function formatValue(value) {
    value = value.trim();

    // Если это массив или объект, форматируем с отступами
    if ((value.startsWith('[') && value.endsWith(']')) ||
        (value.startsWith('{') && value.endsWith('}'))) {
        try {
            const parsed = JSON.parse(value);
            return JSON.stringify(parsed, null, 2);
        } catch (e) {
            return value;
        }
    }

    return value;
}

// Экранирует HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Show "no data" message
function showNoData() {
    document.getElementById('content').innerHTML = `
                <div class="no-data">
                    <h2>No evaluation results available</h2>
                    <p>Run an evaluation with <code>show_dashboard=True</code> to see results here.</p>
                </div>
            `;
}

// Render dashboard
function renderDashboard(session) {
    try {
        const data = session.data;

        if (!data || !data.test_cases || !data.metrics_summary) {
            showNoData();
            return;
        }

        document.getElementById('timestamp').textContent = `Generated: ${session.timestamp || 'Unknown'}`;

        const metricsLabels = Object.keys(data.metrics_summary);
        const metricsScores = metricsLabels.map(m => data.metrics_summary[m].avg_score || 0);

        // Prepare data for pass/fail chart
        const metricsPassed = metricsLabels.map(m => data.metrics_summary[m].passed || 0);
        const metricsFailed = metricsLabels.map(m => data.metrics_summary[m].failed || 0);

        let metricCards = '';
        for (const [metricName, metricData] of Object.entries(data.metrics_summary)) {
            // Определяем класс цвета в зависимости от threshold
            const scoreClass = (metricData.avg_score || 0) >= (metricData.threshold || 0) ? 'passed' : 'failed';

            metricCards += `
            <div class="metric-card">
                <h3>${metricName}</h3>
                <div class="metric-score ${scoreClass}">${(metricData.avg_score || 0).toFixed(3)}</div>
                <div class="metric-details">
                    <p><strong>Passed:</strong> ${metricData.passed || 0}</p>
                    <p><strong>Failed:</strong> ${metricData.failed || 0}</p>
                    <p><strong>Success Rate:</strong> ${(metricData.success_rate || 0).toFixed(1)}%</p>
                    <p><strong>Threshold:</strong> ${metricData.threshold || 'N/A'}</p>
                    <p><strong>Model:</strong> ${metricData.model || 'N/A'}</p>
                    <p><strong>Cost:</strong> $${(metricData.total_cost || 0).toFixed(6)}</p>
                </div>
            </div>
        `;
        }

        let tableRows = '';
        data.test_cases.forEach((testCase, idx) => {
            const metrics = testCase.metrics || [];
            const overallStatus = metrics.length > 0 && metrics.every(m => m.success) ? 'passed' : 'failed';
            const statusBadge = overallStatus === 'passed'
                ? '<span class="status-badge passed">PASSED</span>'
                : '<span class="status-badge failed">FAILED</span>';

            const inputHtml = renderMarkdown(testCase.input || 'N/A');
            const outputHtml = renderMarkdown(testCase.actual_output || 'N/A');

            tableRows += `
            <tr>
                <td class="test-number">${idx + 1}</td>
                <td>${statusBadge}</td>
                <td class="query-cell"><div class="markdown-content">${inputHtml}</div></td>
                <td class="response-cell"><div class="markdown-content">${outputHtml}</div></td>
                <td>
                    <button class="view-details-btn" onclick="showDetails(${idx})">
                        Details
                    </button>
                </td>
            </tr>
        `;
        });

        document.getElementById('content').innerHTML = `
                <div class="summary">
                    <div class="summary-card">
                        <h3>Total Tests</h3>
                        <div class="value">${data.total_tests || 0}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Total Cost</h3>
                        <div class="value">$${(data.total_cost || 0).toFixed(6)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Metrics</h3>
                        <div class="value">${metricsLabels.length}</div>
                    </div>
                </div>
                
                <h2 class="section-title">Metrics Summary</h2>
                <div class="metrics-grid">
                    ${metricCards}
                </div>
                
                <h2 class="section-title">Charts</h2>
                <div class="charts">
                    <div class="chart-container">
                        <h2>Average Scores by Metric</h2>
                        <canvas id="scoresChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h2>Pass/Fail by Metric</h2>
                        <canvas id="passFailChart"></canvas>
                    </div>
                </div>
                
                <h2 class="section-title">Test Results</h2>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 60px;">#</th>
                                <th style="width: 100px;">Status</th>
                                <th>Input</th>
                                <th>Actual Output</th>
                                <th style="width: 120px;">Details</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${tableRows}
                        </tbody>
                    </table>
                </div>
            `;

        renderCharts(metricsLabels, metricsScores, metricsPassed, metricsFailed);
    } catch (error) {
        console.error('Error rendering dashboard:', error);
        showNoData();
    }
}

// Show details in modal
function showDetails(testCaseIdx) {
    try {
        if (!currentData || !currentData.test_cases || !currentData.test_cases[testCaseIdx]) {
            return;
        }

        const testCase = currentData.test_cases[testCaseIdx];
        const metrics = testCase.metrics || [];
        const overallStatus = metrics.length > 0 && metrics.every(m => m.success) ? 'passed' : 'failed';
        const statusBadge = overallStatus === 'passed'
            ? '<span class="status-badge passed">PASSED</span>'
            : '<span class="status-badge failed">FAILED</span>';

        document.getElementById('modalTitle').innerHTML = `Test Case: #${testCaseIdx + 1} ${statusBadge}`;

        // Build metrics panel
        let metricsHtml = '';
        metrics.forEach((metric, idx) => {
            const statusClass = metric.success ? 'passed' : 'failed';
            const icon = metric.success ? '✓' : '✗';

            const fullReason = metric.reason_full || metric.reason || '';

            metricsHtml += `
                    <div class="metric-result-item ${statusClass}">
                        <div class="metric-result-header">
                            <div class="metric-name">
                                <span class="metric-icon ${statusClass}">${icon}</span>
                                ${metric.name}
                            </div>
                            <div class="metric-score-display ${statusClass}">
                                ${metric.score.toFixed(2)}
                            </div>
                        </div>
                        <div class="metric-explanation">
                            <strong>Explanation:</strong> ${fullReason}
                        </div>
                        <div class="metric-meta">
                            <span>Eval cost: $${(metric.evaluation_cost || 0).toFixed(6)}</span>
                            <span>Threshold: ${metric.threshold}</span>
                        </div>
                        <button class="metric-view-details" onclick="showMetricDetails(${testCaseIdx}, ${idx})">
                            View Details
                        </button>
                    </div>
                `;
        });

        // Build main content
        let modalContent = `
            <div class="detail-layout">
                <div class="detail-main">
                    <div class="detail-section">
                        <h3>INPUT</h3>
                        <div class="collapsible-content ${(testCase.input_full || testCase.input).length > 500 ? '' : 'expanded'}" id="input-content">
                            <div class="content markdown-content">${renderMarkdown(testCase.input_full || testCase.input)}</div>
                        </div>
                        ${(testCase.input_full || testCase.input).length > 500 ? '<span class="see-all" data-toggle="input-content">See all</span>' : ''}
                    </div>
                    
                    <div class="detail-section">
                        <h3>ACTUAL OUTPUT</h3>
                        <div class="collapsible-content ${(testCase.actual_output_full || testCase.actual_output || '').length > 500 ? '' : 'expanded'}" id="response-content">
                            <div class="content markdown-content">${renderMarkdown(testCase.actual_output_full || testCase.actual_output || 'N/A')}</div>
                        </div>
                        ${(testCase.actual_output_full || testCase.actual_output || '').length > 500 ? '<span class="see-all" data-toggle="response-content">See all</span>' : ''}
                    </div>
        `;
        // Add context if available - РАЗДЕЛИТЬ НА ЧАНКИ
        if (testCase.retrieval_context && testCase.retrieval_context.length > 0) {
            // Вычисляем общую длину всех чанков
            const totalContextLength = testCase.retrieval_context.join('').length;
            const contextNeedsCollapse = totalContextLength > 1000;

            let contextHtml = '';
            testCase.retrieval_context.forEach((chunk, idx) => {
                const chunkId = `context-chunk-${idx}`;
                const chunkNeedsCollapse = chunk.length > 500;

                contextHtml += `
            <div class="context-chunk">
                <div class="context-chunk-header">Chunk ${idx + 1}</div>
                <div class="collapsible-content ${chunkNeedsCollapse ? '' : 'expanded'}" id="${chunkId}">
                    <div class="context-chunk-content markdown-content">${renderMarkdown(chunk)}</div>
                </div>
                ${chunkNeedsCollapse ? `<span class="see-all" data-toggle="${chunkId}">See all</span>` : ''}
            </div>
        `;
            });

            modalContent += `
            <div class="detail-section">
                <h3>CONTEXT</h3>
                <div class="collapsible-content ${contextNeedsCollapse ? '' : 'expanded'}" id="all-context-content">
                    <div class="content">
                        ${contextHtml}
                    </div>
                </div>
                ${contextNeedsCollapse ? '<span class="see-all" data-toggle="all-context-content">See all context</span>' : ''}
            </div>
        `;
        }

        // Add expected output if available
        if (testCase.expected_output_full || testCase.expected_output) {
            const expectedText = testCase.expected_output_full || testCase.expected_output;
            modalContent += `
            <div class="detail-section">
                <h3>EXPECTED ANSWER</h3>
                <div class="collapsible-content ${expectedText.length > 500 ? '' : 'expanded'}" id="expected-content">
                    <div class="content markdown-content">${renderMarkdown(expectedText)}</div>
                </div>
                ${expectedText.length > 500 ? '<span class="see-all" data-toggle="expected-content">See all</span>' : ''}
            </div>
        `;
        }

        // Add metadata sections
        modalContent += `
                        <div class="detail-section">
                            <h3>RESPONSE TIME</h3>
                            <div class="content">
                                <div class="info-row">
                                    <span class="info-label">Duration:</span>
                                    <span class="info-value">${testCase.response_time || 'N/A'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metrics-panel">
                        <div class="detail-section" style="margin-bottom: 16px;">
                            <h3>METRICS</h3>
                        </div>
                        ${metricsHtml}
                    </div>
                </div>
            `;

        document.getElementById('modalBody').innerHTML = modalContent;
        document.getElementById('modalBody').addEventListener('click', function (e) {
            if (e.target.classList.contains('see-all')) {
                const targetId = e.target.getAttribute('data-toggle');
                if (targetId) {
                    toggleCollapse(targetId);
                }
            }
        });

        document.getElementById('detailsModal').style.display = 'block';
    } catch (error) {
        console.error('Error showing details:', error);
    }
}

// Close modal
function closeModal() {
    document.getElementById('detailsModal').style.display = 'none';
}

// Close on click outside modal
window.onclick = function (event) {
    const modal = document.getElementById('detailsModal');
    if (event.target == modal) {
        closeModal();
    }
}

// Render charts
function renderCharts(labels, scores, passed, failed) {
    if (scoresChart) scoresChart.destroy();
    if (passFailChart) passFailChart.destroy();

    const scoresCtx = document.getElementById('scoresChart').getContext('2d');
    scoresChart = new Chart(scoresCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average Score',
                data: scores,
                backgroundColor: '#E88213',
                borderColor: '#E88213',
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: {
                        color: '#2a3f5f'
                    },
                    ticks: {
                        color: '#888'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#888'
                    }
                }
            }
        }
    });

    const passFailCtx = document.getElementById('passFailChart').getContext('2d');
    passFailChart = new Chart(passFailCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Passed',
                    data: passed,
                    backgroundColor: '#43A047',
                    borderRadius: 4
                },
                {
                    label: 'Failed',
                    data: failed,
                    backgroundColor: '#ef5350',
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        color: '#888'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    stacked: false,
                    grid: {
                        color: '#2a3f5f'
                    },
                    ticks: {
                        color: '#888',
                        precision: 0
                    }
                },
                x: {
                    stacked: false,
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#888'
                    }
                }
            }
        }
    });
}

// Refresh data
function refreshData() {
    loadSessions();
    loadSession();
}

// Clear cache
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

function showMetricDetails(testCaseIdx, metricIdx) {
    try {
        if (!currentData || !currentData.test_cases || !currentData.test_cases[testCaseIdx]) {
            return;
        }

        const testCase = currentData.test_cases[testCaseIdx];
        const metric = testCase.metrics[metricIdx];

        if (!metric) return;

        const statusClass = metric.success ? 'passed' : 'failed';
        const statusBadge = metric.success
            ? '<span class="status-badge passed">PASSED</span>'
            : '<span class="status-badge failed">FAILED</span>';

        // Create modal if doesn't exist
        let metricModal = document.getElementById('metricDetailsModal');
        if (!metricModal) {
            metricModal = document.createElement('div');
            metricModal.id = 'metricDetailsModal';
            metricModal.className = 'metric-detail-modal';
            metricModal.innerHTML = `
                <div class="metric-detail-content">
                    <div class="modal-header">
                        <h2 id="metricModalTitle">Metric Details</h2>
                        <span class="close" onclick="closeMetricModal()">&times;</span>
                    </div>
                    <div class="metric-detail-body" id="metricModalBody"></div>
                </div>
            `;
            document.body.appendChild(metricModal);
        }

        document.getElementById('metricModalTitle').innerHTML = `${metric.name} ${statusBadge}`;

        let detailsHtml = `
            <div class="metric-detail-section">
                <h4>SCORE</h4>
                <div class="content">
                    <div class="info-row">
                        <span class="info-label">Score:</span>
                        <span class="info-value">${metric.score.toFixed(3)}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Threshold:</span>
                        <span class="info-value">${metric.threshold}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Status:</span>
                        <span class="info-value">${metric.success ? 'Passed' : 'Failed'}</span>
                    </div>
                </div>
            </div>

            <div class="metric-detail-section">
                <h4>EXPLANATION</h4>
                <div class="content markdown-content">
                    ${renderMarkdown(metric.reason_full || metric.reason || 'No explanation available')}
                </div>
            </div>

            <div class="metric-detail-section">
                <h4>EVALUATION INFO</h4>
                <div class="content">
                    <div class="info-row">
                        <span class="info-label">Model:</span>
                        <span class="info-value">${metric.evaluation_model || 'N/A'}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Cost:</span>
                        <span class="info-value">$${(metric.evaluation_cost || 0).toFixed(6)}</span>
                    </div>
                </div>
            </div>
        `;

        // Add evaluation log if available
        if (metric.evaluation_log) {
            // Вычисляем индекс лога (testCaseIdx * кол-во метрик + metricIdx)
            const logIndex = testCaseIdx * (currentData.test_cases[testCaseIdx].metrics.length) + metricIdx;

            // Берем RAW строку из хранилища
            const rawLogStr = rawEvaluationLogs.get(metricIdx) || rawEvaluationLogs.get(logIndex);

            console.log(`=== Показываем лог для test=${testCaseIdx}, metric=${metricIdx}, index=${logIndex} ===`);
            console.log('rawLogStr найден:', !!rawLogStr);

            let formattedLog;
            if (rawLogStr) {
                formattedLog = formatEvaluationLogFromRaw(rawLogStr);
            } else {
                formattedLog = 'Log not found in RAW text.';
            }

            detailsHtml += `
        <div class="metric-detail-section">
            <h4>EVALUATION LOG</h4>
            <div class="content">
                <pre class="metric-detail-log">${formattedLog}</pre>
            </div>
        </div>
    `;
        }

        document.getElementById('metricModalBody').innerHTML = detailsHtml;
        metricModal.style.display = 'block';
    } catch (error) {
        console.error('Error showing metric details:', error);
    }
}
// Close metric detail modal
function closeMetricModal() {
    const modal = document.getElementById('metricDetailsModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Close metric modal on outside click
window.addEventListener('click', function (event) {
    const modal = document.getElementById('metricDetailsModal');
    if (modal && event.target == modal) {
        closeMetricModal();
    }
});


// Add event listener for session selection
document.addEventListener('DOMContentLoaded', function () {
    const sessionSelect = document.getElementById('sessionSelect');
    if (sessionSelect) {
        sessionSelect.addEventListener('change', loadSession);
    }

    // Initialize
    loadSessions();
    loadSession();
});