import networkx as nx
import pandas as pd

def build_entity_graph(edges_df):
    g = nx.Graph()
    for _, r in edges_df.iterrows():
        g.add_edge(r['src'], r['dst'], kind=r['kind'])
    return g

def risky_components(g, seed_nodes):
    risky = set(seed_nodes)
    for n in seed_nodes:
        if n in g:
            risky.update(g.neighbors(n))
    return risky

if __name__ == '__main__':
    edges = pd.DataFrame({
        'src': ['C1','C1','C2','D1'],
        'dst': ['D1','A1','D1','M1'],
        'kind': ['uses_device','owns_account','uses_device','merchant'],
    })
    g = build_entity_graph(edges)
    print(nx.number_of_nodes(g), nx.number_of_edges(g))
    print(risky_components(g, ['C1']))
