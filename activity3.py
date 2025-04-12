import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class MLRouteSystem:
    def __init__(self):
        self.graph = nx.Graph()
        self.max_transfers = 2
        self.max_time = 120
        self.model = None

    def add_station(self, name, coords=(0, 0)):
        self.graph.add_node(name, pos=coords)

    def add_connection(self, origin, dest, distance, time, route_name):
        if time <= self.max_time:
            self.graph.add_edge(
                origin, dest, distance=distance, time=time, route_name=route_name
            )
    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)

        data = []
        for _, row in df.iterrows():
            origin = row["origen_ruta_troncal"]
            dest = row["destino_ruta_troncal"]
            distance = row["longitud_ruta_troncal"]
            route_name = row["route_name_ruta_troncal"]
            time = (distance / 20) * 60

            if not self.graph.has_node(origin):
                self.add_station(origin)
            if not self.graph.has_node(dest):
                self.add_station(dest)

            self.add_connection(origin, dest, distance, time, route_name)
            data.append([distance, time])

        X = np.array([d[0] for d in data]).reshape(-1, 1)
        y = np.array([d[1] for d in data])
        self.train_model(X, y)

        return df

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model.fit(X_train, y_train)
    def find_route(self, start, end):
        if self.model:
            for u, v, data in self.graph.edges(data=True):
                distance = data["distance"]
                data["time"] = self.model.predict([[distance]])[0]

        route = nx.shortest_path(self.graph, start, end, weight="time")
        return route if len(route) - 1 <= self.max_transfers else []

    def analyze_route(self, route):
        if not route or len(route) < 2:
            return {"route": [], "distance": 0, "time": 0, "route_names": []}

        total_distance = total_time = 0
        route_names = []

        for i in range(len(route) - 1):
            edge_data = self.graph.get_edge_data(route[i], route[i + 1])
            total_distance += edge_data["distance"]
            total_time += edge_data["time"]
            route_names.append(edge_data["route_name"])

        return {
            "route": route,
            "distance": total_distance,
            "time": total_time,
            "route_names": route_names,
        }

system = MLRouteSystem()
csv_path = "Rutas_Troncales_de_TRANSMILENIO.csv"
df = system.load_data(csv_path)

origin = "PORTAL NORTE"
dest = "SAN MATEO - CC UNISUR"

route = system.find_route(origin, dest)
details = system.analyze_route(route)

print(f"\nOptimal route from {origin} to {dest}:")
print(f"Stations: {' → '.join(details['route'])}")

print("\nDetailed route:")
for i in range(len(details["route"]) - 1):
    print(
        f"{details['route'][i]} → {details['route'][i+1]} (Route: {details['route_names'][i]})"
    )

print(f"\nTotal distance: {details['distance']:.2f} km")
print(f"Estimated time: {details['time']:.1f} minutes")