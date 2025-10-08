# ----------------------------------------------------------------------
# Impor Semua Library
# ----------------------------------------------------------------------
import geopandas as gpd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString
import networkx as nx
import random
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------
# Definisikan Semua Konstanta & Path File di Atas
# ----------------------------------------------------------------------
CRS_PROYEKSI = "EPSG:32749"
OUTPUT_DIR = "hasil"
PATH_JALAN_MENTAH = "jalan_surabaya.shp"
PATH_GENANGAN = "genangan.shp"
PATH_EVAKUASI = "Evakuasi.shp"
KOORDINAT_AWAL_LATLON = (-7.30479, 112.72719)
NAMA_TUJUAN = "Lapangan Kodam"

# ----------------------------------------------------------------------
# Fungsi 1: Proses KNN
# ----------------------------------------------------------------------
def jalankan_knn():
    print("--- Memulai Proses KNN ---")
    
    try:
        jalan = gpd.read_file(PATH_JALAN_MENTAH)
        genangan = gpd.read_file(PATH_GENANGAN)
    except Exception as e:
        print(f"FATAL: Gagal membaca file mentah. Pastikan path di konstanta sudah benar. Detail: {e}")
        return None

    jalan = jalan.to_crs(CRS_PROYEKSI)
    genangan = genangan.to_crs(CRS_PROYEKSI)

    print("Menghitung fitur jarak ke genangan...")
    jarak_list = [geom.distance(nearest_points(geom, genangan.unary_union)[1]) for geom in jalan.geometry]
    jalan['jarak_banjir'] = jarak_list

    labels = [2 if jarak < 50 else 1 if jarak < 200 else 0 for jarak in jalan['jarak_banjir']]
    jalan['risiko'] = labels

    print("Melatih model KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(jalan[['jarak_banjir']], jalan['risiko'])
    jalan['risiko_prediksi'] = knn.predict(jalan[['jarak_banjir']])
    
    print("--- Proses KNN Selesai ---")
    return jalan

# ----------------------------------------------------------------------
# Fungsi 2: Proses RL
# ----------------------------------------------------------------------
def jalankan_rl(jalan_berisiko):
    print("\n--- Memulai Proses RL ---")
    
    if jalan_berisiko is None:
        print("FATAL: Proses RL dibatalkan karena data dari KNN tidak valid.")
        return

    # --- Persiapan Titik Awal & Akhir ---
    print("Menyiapkan Titik Awal dan Tujuan...")
    # ... (Kode titik awal dan akhir tetap sama)
    lat_awal, lon_awal = KOORDINAT_AWAL_LATLON
    gdf_awal = gpd.GeoDataFrame([1], geometry=[Point(lon_awal, lat_awal)], crs="EPSG:4326")
    gdf_awal_utm = gdf_awal.to_crs(CRS_PROYEKSI)
    start_coord = (gdf_awal_utm.geometry.iloc[0].x, gdf_awal_utm.geometry.iloc[0].y)

    evakuasi = gpd.read_file(PATH_EVAKUASI).to_crs(CRS_PROYEKSI)
    titik_tujuan = evakuasi[evakuasi['lokasi'] == NAMA_TUJUAN]
    if titik_tujuan.empty:
        raise ValueError(f"Lokasi tujuan '{NAMA_TUJUAN}' tidak ditemukan di file {PATH_EVAKUASI}")
    end_coord = (titik_tujuan.geometry.iloc[0].x, titik_tujuan.geometry.iloc[0].y)
    
    # --- Pembangunan Graf ---
    print("Membangun Graf Jaringan Jalan...")
    G = nx.Graph()
    for _, row in jalan_berisiko.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords) - 1):
            G.add_edge(coords[i], coords[i+1], risk=row['risiko_prediksi'], length=Point(coords[i]).distance(Point(coords[i+1])))
    
    nodes = np.array(list(G.nodes))
    start_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - start_coord, axis=1))])
    end_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - end_coord, axis=1))])
    
        # Di dalam fungsi jalankan_rl
    # ... setelah baris end_node = tuple(...)

    print(f"Agen akan berjalan dari {start_node} ke {end_node}")

    # Cek 1: Apakah kedua titik ada di dalam graf jalan?
    if not G.has_node(start_node) or not G.has_node(end_node):
        print("FATAL: Titik awal atau akhir tidak ditemukan di jaringan jalan.")
        return

    # Cek 2: Apakah ada jalur fisik yang menghubungkan keduanya?
    if not nx.has_path(G, start_node, end_node):
        print("FATAL: Tidak ada jalur fisik yang menghubungkan titik awal dan akhir di graf.")
        return

    # --- Algoritma Q-Learning ---
    print("Melatih Agen RL...")
    q_table = {node: {neighbor: 0 for neighbor in G.neighbors(node)} for node in G.nodes()}
    alpha, gamma = 0.1, 0.6
    
    for i in range(20000): # episodes
        state = start_node
        for _ in range(500): # max steps
            if state == end_node: break
            
            action = max(q_table[state], key=q_table[state].get) if random.uniform(0, 1) > 0.2 else random.choice(list(G.neighbors(state)))
            risk, length = G[state][action]['risk'], G[state][action]['length']
            reward = -length - (500 if risk == 2 else 200 if risk == 1 else 0)
            next_state = action
            
            next_max = max(q_table[next_state].values()) if q_table.get(next_state) else 0
            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * next_max)
            state = next_state
    
    # --- Ekstraksi Rute & Penyimpanan Hasil ---
    print("Mengekstrak rute terbaik...")
    path = [start_node]
    while path[-1] != end_node:
        if not q_table.get(path[-1]): path = []; break
        path.append(max(q_table[path[-1]], key=q_table[path[-1]].get))
        if len(path) > len(G.nodes()): path = []; break
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Membuat GeoDataFrame dari path yang ditemukan
    if path:
        rute_gdf = gpd.GeoDataFrame(geometry=[LineString(path)], crs=CRS_PROYEKSI)
        rute_gdf.to_file(os.path.join(OUTPUT_DIR, "rute_evakuasi_final.shp"))
        print(f"Berhasil! Rute disimpan di folder '{OUTPUT_DIR}'.")
    else:
        print("Gagal menemukan rute.")
    
    print("--- Proses RL Selesai ---")

# ----------------------------------------------------------------------
# Bagian Eksekusi Utama
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Memulai Alur Kerja Pencarian Rute Evakuasi...")
    
    data_hasil_knn = jalankan_knn()
    
    if data_hasil_knn is not None:
        # Simpan hasil KNN ke file untuk verifikasi
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        data_hasil_knn.to_file(os.path.join(OUTPUT_DIR, "jalan_dengan_risiko.shp"))
        print(f"File 'jalan_dengan_risiko.shp' telah disimpan di folder '{OUTPUT_DIR}'.")
        
        # Jalankan RL menggunakan hasil dari KNN
        jalankan_rl(data_hasil_knn)

    print("\nSkrip telah selesai dijalankan.")