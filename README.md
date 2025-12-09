# LAPORAN  
## PROYEK MATA KULIAH DEEP LEARNING  
### “…..”

---

## **Disusun oleh Kelompok 1**
- Tsabita Rosyidah Putri — (22083010012)  
- Adelia Yuandhika — (22083010066)  
- Hazza Fitrah Hafizhul Chaq — (22083010070)  
- Niken Sulistyowati — (22083010091)  
- Farah Yusnaida Arif — (22083010106)

### **Dosen Pengampu**
Amri Muhaimin, S.Stat., M.Stat., MS  
NIP. 199507232024061002

### **Program Studi Sains Data**  
**Fakultas Ilmu Komputer**  
Universitas Pembangunan Nasional “Veteran” Jawa Timur  
2025

---

# **BAB I — PENDAHULUAN**

## **Latar Belakang**
Dalam bidang kecerdasan buatan (Artificial Intelligence), deep learning merupakan metode yang populer untuk pemrosesan citra, khususnya dengan memanfaatkan kemampuan jaringan saraf tiruan untuk mengekstrak fitur dari data berdimensi tinggi. Hal ini memicu pengembangan model generatif yang bertujuan untuk mempelajari representasi data sehingga sistem mampu menghasilkan sampel baru yang relevan. Salah satu arsitektur generatif yang banyak digunakan adalah **Autoencoder (AE)**.

Autoencoder (AE) dapat mengubah data berukuran besar menjadi representasi yang lebih kecil tanpa kehilangan informasi penting. Metode ini banyak dimanfaatkan untuk:
- ekstraksi fitur secara otomatis,
- kompresi data visual,
- deteksi anomali.

AE berfungsi untuk mengekstrak informasi data, yaitu variabel laten, serta membuang _noise_ yang tidak diperlukan. AE bekerja dengan mengkompresi data input menjadi representasi berdimensi rendah melalui *encoder*, kemudian merekonstruksi kembali data menggunakan *decoder*. Struktur utama AE terdiri dari:
- **Encoder** → mengekstrak variabel laten,  
- **Bottleneck** → ruang laten tempat informasi dipadatkan,  
- **Decoder** → merekonstruksi kembali data ke bentuk semula.

AE memiliki kemampuan untuk mengompres data ke dalam ruang laten berdimensi rendah. Namun, representasi laten yang dihasilkan AE sering tidak terstruktur sehingga keterampilan model dalam menghasilkan sampel baru menjadi terbatas. Hal ini dapat diatasi dengan **Variational Autoencoder (VAE)**.

Susanto dan Pardede membandingkan penggunaan AE dan VAE untuk reduksi dimensi pada prediksi cacat mesin mobil. Hasil penelitian menunjukkan bahwa VAE menghasilkan kinerja yang lebih baik dibandingkan AE.

Arsitektur VAE diperkenalkan oleh Dienderik P. Kingma dan Max Welling melalui makalah Auto-Encoding Variational Bayes pada tahun 2013. Variational Autoencoder (VAE) merupakan model generatif probabilistik yang mengkodekan variabel laten sebagai distribusi probabilitas. VAE mengestimasi dua vektor, antara lain:
- **rata-rata (μ)**  
- **standar deviasi (σ)**  

Pendekatan ini memungkinkan terbentuknya ruang laten yang terstruktur, mulus, dan dapat diinterpolasi. Fungsi loss VAE terdiri dari:
- **Reconstruction Loss**  
- **Kullback-Leibler Divergence**

Nugroho dkk menggunakan VAE untuk mengekstraksi fitur pada sistem deteksi api. HAsilnya menunjukkan bahwa vektor laten mengalami perubahan menjadi lebih kecil dibandingkan dengan data awal tanpa kehilangan fitur penting. Disisi lain Giger dan Csillaghy menggunakan VAE untuk mendeteksi anomali pada _full-disk solar images_. 

Berbagai penelitian sebelumnya menunjukkan efektivitas VAE dalam deteksi anomali pada citra terstruktur seperti inspeksi industri, api, dan citra matahari. Namun, citra wajah jauh lebih kompleks karena variasi ekspresi, pose, pencahayaan, serta tekstur kulit sehingga representasi laten menjadi lebih kompleks. Dalam pemrosesan wajah, VAE memiliki kemampuan untuk mempelajari representasi laten yang terpisah sehingga fitur seperti warna rambut, ekspresi, bentuk wajah, dan tekstur dapat dimanipulasi secara independen.

Terkait dengan kinerja VAE yang digunakan untuk kompresi vektor laten yang diterapkan pada data citra wajah yang memiliki representasi laten yang kompleks. _Output_ yang dihasilkan dapat cenderung memiliki _noise_ atau mengalami blur. Permasalahan tersebut dapat diatasi dengan menggunakan **_Residual Blocks_** yang dapat membantu model mempertahankan informasi penting selama proses encoding dan decoding. Dengan demikian, VAE dapat menghasilkan rekonstruksi wajah yang lebih tajam dan mengurangi _noise_ atau blur karena hilangnya informasi sehingga dapat meningkatkan kualitas _output_.


Proyek ini menerapkan VAE dengan menambahkan arsitektur _Residual Blocks_ pada **CelebA Dataset**, sebuah dataset citra wajah populer dengan variasi atribut yang tinggi. Kompleksitas dataset ini menjadikannya uji coba ideal untuk mengevaluasi kemampuan VAE dengan Residual Blocks dalam rekonstruksi dan generasi citra wajah.

---

## **Rumusan Masalah**
1. Bagaimana mengimplementasikan Variational Autoencoder (VAE) dengan Residual Blocks dalam mempelajari representasi fitur wajah pada dataset CelebA?  
2. Bagaimana kinerja Variational Autoencoder (VAE) dengan Residual Blocks dalam merekonstruksi citra wajah dari dataset CelebA?  
3. Seberapa realistis dan beragam citra sintetis yang dapat dihasilkan oleh Variational Autoencoder (VAE) dengan Residual Blocks terhadap data CelebA?

---

## **Tujuan Penelitian**
1. Mengimplementasikan Variational Autoencoder (VAE) dengan Residual Blocks untuk dataset CelebA.  
2. Menganalisis kinerja rekonstruksi model VAE dengan Residual Blocks pada data wajah CelebA.  
3. Mendemonstrasikan citra wajah sintetis yang dihasilkan VAE dengan Residual Blocks serta mengevaluasi keragaman dan realismenya.

---

**`| Header 1 | Header 2 |`**
**`:---`, `:---:`, `---:`**
**`| Sel 1 | Sel 2 |`**
