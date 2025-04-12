import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class UnsupervisedRouteSystem:
    def __init__(self):
        self.graph = nx.Graph()
        self.max_transfers = 2
        self.max_time = 200
        self.clusters = None
        self.scaler = StandardScaler()
        self.route_clusters = {}

    def add_station(self, name, coords=(0, 0)):
        self.graph.add_node(name, pos=coords)

    def add_connection(self, origin, dest, distance, time, route_name):
        if time <= self.max_time:
            self.graph.add_edge(
                origin, dest, distance=distance, time=time, route_name=route_name
            )
    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)

        features = []
        route_data = []

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

            features.append([distance, time])
            route_data.append((origin, dest, route_name))
        self.apply_clustering(features, route_data)

        return df

    def apply_clustering(self, features, route_data):
        X = self.scaler.fit_transform(features)

        n_clusters = min(5, len(X))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        for i, (origin, dest, route_name) in enumerate(route_data):
            key = (origin, dest)
            self.route_clusters[key] = clusters[i]

            self.graph[origin][dest]["cluster"] = clusters[i]

            cluster_weight = (clusters[i] + 1) * 0.5
            self.graph[origin][dest]["time"] *= cluster_weight

        print(
            f"Routes grouped into {n_clusters} clusters based on distance and time patterns"
        )
    def find_route(self, start, end):
        route = nx.shortest_path(self.graph, start, end, weight="time")
        return route if len(route) - 1 <= self.max_transfers else []

    def analyze_route(self, route):
        if not route or len(route) < 2:
            return {
                "route": [],
                "distance": 0,
                "time": 0,
                "route_names": [],
                "clusters": [],
            }

        total_distance = total_time = 0
        route_names = []
        clusters = []

        for i in range(len(route) - 1):
            edge_data = self.graph.get_edge_data(route[i], route[i + 1])
            total_distance += edge_data["distance"]
            total_time += edge_data["time"]
            route_names.append(edge_data["route_name"])
            clusters.append(edge_data["cluster"])

        return {
            "route": route,
            "distance": total_distance,
            "time": total_time,
            "route_names": route_names,
            "clusters": clusters,
        }
system = UnsupervisedRouteSystem()
csv_path = "Rutas_Troncales_de_TRANSMILENIO.csv"
df = system.load_data(csv_path)

origin = "MUSEO NACIONAL - FNG"
dest = "PORTAL NORTE"

route = system.find_route(origin, dest)
details = system.analyze_route(route)

print(f"\nOptimal route from {origin} to {dest}:")
print(f"Stations: {' → '.join(details['route'])}")

print("\nDetailed route:")
for i in range(len(details["route"]) - 1):
    print(
        f"{details['route'][i]} → {details['route'][i+1]} (Route: {details['route_names'][i]}, Cluster: {details['clusters'][i]})"
    )

print(f"\nTotal distance: {details['distance']:.2f} km")
print(f"Estimated time: {details['time']:.1f} minutes")
