# Heatmap
st.subheader("Heatmap")

st.markdown("""
**Penjelasan Warna pada Heatmap:**

- **Warna Merah Tua / Kuning Tua:**  
  Menunjukkan daerah dengan curah hujan yang tinggi. Lokasi-lokasi yang lebih intens curah hujannya akan tampak dengan warna yang lebih gelap. Daerah dengan intensitas curah hujan tinggi sering kali berwarna merah tua atau oranye terang, menunjukkan curah hujan yang sangat tinggi.

- **Warna Kuning / Hijau Muda:**  
  Menunjukkan daerah dengan curah hujan sedang. Warna-warna seperti kuning atau hijau muda menandakan intensitas hujan yang lebih rendah dibandingkan dengan daerah merah.

- **Warna Biru Tua / Biru Muda:**  
  Menunjukkan daerah dengan curah hujan rendah. Ini biasanya mewakili lokasi-lokasi yang memiliki sedikit atau bahkan tidak ada hujan (seperti musim kemarau). Warna biru gelap atau biru muda ini menandakan intensitas hujan yang sangat rendah.
""")

heatmap = create_heatmap(df)
st_folium(heatmap, width=700, height=500)

# Distribusi Cluster per Kabupaten
st.subheader("Distribusi Cluster per Kabupaten")

st.markdown("""
**Penjelasan Cluster Berdasarkan Curah Hujan Pada Distribusi Cluster per Kabupaten:**

- **Cluster 0 (Curah Hujan Tinggi - Musim Hujan):**  
  Cluster ini menunjukkan daerah-daerah yang mengalami curah hujan tinggi. Biasanya cluster ini mewakili wilayah yang terletak di musim hujan atau daerah dengan iklim tropis yang sering mengalami hujan deras.  
  **Ciri-ciri:** Area yang termasuk dalam cluster ini akan menunjukkan intensitas curah hujan yang lebih tinggi (lebih dari rata-rata), yang biasanya terkait dengan musim hujan.

- **Cluster 2 (Curah Hujan Sedang - Cuaca Normal):**  
  Cluster ini berisi daerah-daerah dengan curah hujan sedang, yang biasanya terjadi pada cuaca normal atau musim transisi antara musim hujan dan kemarau.  
  **Ciri-ciri:** Wilayah yang termasuk dalam cluster ini memiliki tingkat curah hujan yang cukup stabil, tidak terlalu tinggi dan juga tidak terlalu rendah, mencerminkan cuaca yang tidak ekstrem.

- **Cluster 1 (Curah Hujan Rendah - Musim Kering):**  
  Cluster ini mencakup daerah-daerah yang mengalami curah hujan rendah, yang biasanya terjadi pada musim kemarau atau wilayah yang lebih kering.  
  **Ciri-ciri:** Area yang termasuk dalam cluster ini cenderung mengalami sedikit hujan atau bahkan tidak ada hujan sama sekali dalam periode tertentu, mencerminkan musim kering atau iklim yang lebih kering.
""")

kota_cluster = df.groupby(['cluster', 'KOTA']).size().reset_index(name='Count')
plt.figure(figsize=(10, 6))
sns.barplot(data=kota_cluster, x='KOTA', y='Count', hue='cluster', palette='viridis')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.title("Distribusi Cluster per Kabupaten", fontsize=14)
plt.xlabel("Kabupaten/Kota", fontsize=12)
plt.ylabel("Jumlah Observasi", fontsize=12)
plt.legend(title="Cluster", fontsize=10, loc='upper right')
st.pyplot(plt)
