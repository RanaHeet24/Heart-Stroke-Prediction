document.addEventListener('DOMContentLoaded', () => {
    const V = 4;
    const INF = 99999;
    const places = ["Delhi", "Mumbai", "Chennai", "Kolkata"];
    
    const graph = [
        [0, 1395, INF, 1553],
        [1395, 0, 1238, INF],
        [INF, 1238, 0, 1366],
        [1553, INF, 1366, 0]
    ];

    const initialMatrixContainer = document.getElementById('initial-matrix');
    const resultMatrixContainer = document.getElementById('result-matrix');
    const calculateBtn = document.getElementById('calculate-btn');

    function createTable(matrix) {
        let table = '<table>';
        // Header row
        table += '<tr><th>From/To</th>';
        places.forEach(place => {
            table += `<th>${place}</th>`;
        });
        table += '</tr>';

        // Data rows
        for (let i = 0; i < V; i++) {
            table += `<tr><td class="header">${places[i]}</td>`;
            for (let j = 0; j < V; j++) {
                if (matrix[i][j] === INF) {
                    table += '<td>INF</td>';
                } else {
                    table += `<td>${matrix[i][j]}</td>`;
                }
            }
            table += '</tr>';
        }
        table += '</table>';
        return table;
    }

    function floydWarshall() {
        const dist = JSON.parse(JSON.stringify(graph)); // Deep copy

        for (let k = 0; k < V; k++) {
            for (let i = 0; i < V; i++) {
                for (let j = 0; j < V; j++) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        return dist;
    }

    // Display initial matrix on page load
    initialMatrixContainer.innerHTML = createTable(graph);

    // Handle button click
    calculateBtn.addEventListener('click', () => {
        const shortestPaths = floydWarshall();
        resultMatrixContainer.innerHTML = createTable(shortestPaths);
    });
});
