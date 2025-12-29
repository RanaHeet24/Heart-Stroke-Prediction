#include <iostream>
using namespace std;

#define V 4
#define INF 99999

void floydWarshall(int graph[V][V], string places[V]) {
    int dist[V][V];
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            dist[i][j] = graph[i][j];
        }
    }
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    cout << "\nAll Pairs Shortest Distance (in km):\n";
    cout << "------------------------------------------------------\n";
    cout << "From/To    ";
    for (int i = 0; i < V; i++) {
        cout << places[i] << "    ";
    }
    cout << "\n------------------------------------------------------\n";

    for (int i = 0; i < V; i++) {
        cout << places[i];
        int space = 10 - places[i].length();
        for (int s = 0; s < space; s++) cout << " ";
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == INF)
                cout << "INF         ";
            else
                cout << dist[i][j] << "        ";
        }
        cout << endl;
    }
    cout << "------------------------------------------------------\n";
    cout << "\nNote: 'INF' means there is no direct or indirect path.\n";
}

int main() {
    string places[V] = {"Delhi", "Mumbai", "Chennai", "Kolkata"};
    int graph[V][V] = {
        {0, 1395, INF, 1553},
        {1395, 0, 1238, INF},
        {INF, 1238, 0, 1366},
        {1553, INF, 1366, 0}
    };

    cout << "   Network Routing Using Floyd’s Algorithm\n";
    cout << "\nCities in the Network:\n";
    for (int i = 0; i < V; i++) {
        cout << i + 1 << ". " << places[i] << endl;
    }

    floydWarshall(graph, places);
    return 0;
}