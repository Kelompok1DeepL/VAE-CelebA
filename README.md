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

Terkait dengan kinerja VAE yang digunakan untuk kompresi vektor laten yang diterapkan pada data citra wajah yang memiliki representasi laten yang kompleks. _Output_ yang dihasilkan dapat cenderung memiliki _noise_ atau mengalami blur. Permasalahan tersebut dapat diatasi dengan menggunakan **_Residual Blocks_** yang dapat membantu model mempertahankan informasi penting selama proses encoding dan decoding, selain itu **_Residual Block_** dapat mengurangi waktu _training data_ karena **_Residual Blocks_** bekerja dengan memberikan _shortcut connection_, yaitu jalur pintas yang memungkinkan sebagian informasi melewati layer tanpa harus diolah. Mekanisme ini membantu mengatasi masalah _vanishing gradient_ sehingga gradien dapat mengalir lebih lancar ke layer awal, membuat proses pembelajaran lebih stabil. Karena model lebih mudah belajar, ia dapat mencapai akurasi atau kualitas rekonstruksi yang baik dalam lebih sedikit langkah pembelajaran. Dengan demikian, VAE dapat menghasilkan rekonstruksi wajah yang lebih tajam dan mengurangi _noise_ atau blur karena hilangnya informasi sehingga dapat meningkatkan kualitas _output_.


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

# **BAB II — TINJAUAN PUSTAKA**

## **2.1 Kerangka Teori**
## **2.1.1 Dataset CelebA**
Dataset _CelebFaces Attributes_ (CelebA) merupakan salah satu dataset wajah yang banyak digunakan dalam bidang computer vision dan deep learning, terutama untuk berbagai aplikasi yang membutuhkan identifikasi wajah dan analisis atribut wajah. Dataset dirancang untuk mendukung beragam penelitian, mulai dari pengenalan ekspresi, pendeteksian atribut seperti seseorang yang tersenyum, memiliki rambut berwarna tertentu, hingga penggunaan kacamata. Gambar-gambar pada CelebA mencakup variasi pose, kondisi latar belakang yang bervariasi, serta individu dari berbagai karakteristik, sehingga sangat cocok untuk proses pelatihan dan pengujian model berbasis citra wajah. Dataset ini awalnya dikembangkan oleh tim penelitian di MMLAB, The Chinese University of Hong Kong.

Secara keseluruhan, CelebA terdiri dari 202.599 gambar wajah selebriti, dengan total 10.177 identitas berbeda, walaupun informasi nama tidak disertakan. Setiap gambar dilengkapi dengan 40 anotasi atribut biner yang menggambarkan karakteristik wajah tertentu serta lima titik landmark yang meliputi posisi kedua mata, hidung, dan dua titik mulut. Dataset ini juga menyediakan berbagai berkas pendukung, seperti kumpulan gambar wajah yang telah melalui proses cropping dan alignment, pembagian data yang direkomendasikan untuk pelatihan, validasi, pengujian, informasi _bounding box_, serta berkas anotasi atribut untuk seluruh gambar.

CelebA digunakan secara luas untuk keperluan penelitian akademik dan tersedia hanya untuk penggunaan non-komersial. Dataset ini telah menjadi dasar bagi banyak studi terkait deteksi dan pengenalan wajah, termasuk penelitian oleh Yang, Luo, Loy, dan Tang (2015) yang mengembangkan pendekatan deteksi wajah berbasis pembelajaran mendalam. Dengan jumlah data yang besar, anotasi yang lengkap, serta keberagaman gambar yang tinggi, CelebA menjadi pilihan ideal untuk membangun dan mengevaluasi model-model yang bertujuan mengenali atribut wajah atau menghasilkan kembali citra wajah secara otomatis.

## **2.1.2 _Deep Learning_**
_Deep Learning_ merupakan bagian dari _Machine Learning_ yang dikembangkan berdasarkan cara kerja jaringan saraf biologis pada otak manusia. Pendekatan ini menggunakan model yang disebut Jaringan Saraf Tiruan (_Artificial Neural Networks_/ANN), yang tersusun dari lapisan-lapisan neuron buatan untuk memproses informasi secara berjenjang. Di dalam _Deep Learning_, terdapat berbagai jenis arsitektur yang dirancang untuk tugas tertentu, seperti _Convolutional Neural Network_ (CNN) untuk pengolahan citra, _Recurrent Neural Network_ (RNN) dan _Long Short-Term Memory_ (LSTM) untuk data berurutan, serta _Self Organizing Maps_ (SOM) untuk pemetaan dan pengelompokan data (Alfarizi M. Riziq Sirfatullah et al., 2023).

## **2.1.3 _Convolutional Neural Network_ (CNN)**
_Convolutional Neural Network_ (CNN) adalah arsitektur _deep learning_ yang dirancang untuk mempelajari pola penting dari data yang memiliki struktur, seperti citra maupun teks. Model ini tersusun atas beberapa jenis lapisan yang bekerja secara bertahap. Bagian konvolusi berfungsi mengekstraksi ciri menggunakan kernel yang bergerak melintasi data input dan menghasilkan _feature map_ yang mewakili pola-pola penting. Setelah itu, lapisan _pooling_ mereduksi ukuran representasi tersebut sehingga model menjadi lebih efisien dan lebih tahan terhadap perubahan posisi atau pergeseran fitur.

Hasil ekstraksi fitur kemudian diratakan (_flattening_) dan diteruskan ke lapisan _fully connected_ untuk proses klasifikasi akhir. Keunggulan utama CNN terletak pada kemampuannya melakukan ekstraksi fitur secara otomatis tanpa memerlukan rekayasa fitur manual, serta sifat _spatial invariance_ yang membuat model tetap mampu mengenali pola meskipun terjadi perubahan posisi atau bentuk kecil pada input. Pendekatan ini menjadikan CNN efektif digunakan dalam berbagai tugas klasifikasi berbasis gambar maupun data teks berurutan (Metlapalli et al., 2020).

## **2.1.4 _Autoencoder_**
_Autoencoder_ merupakan jaringan saraf yang dirancang untuk mempelajari cara merekonstruksi kembali data masukan. Model ini terdiri dari _encoder_ yang memampatkan input menjadi representasi berdimensi rendah, serta _decoder_ yang menghasilkan rekonstruksi dari representasi tersebut. Meskipun mampu menyalin ulang data, nilai utama _autoencoder_ sering terletak pada representasi latennya yang dapat digunakan untuk berbagai tugas analisis (César Pérez Curiel, 2022). Untuk memberikan gambaran visual mengenai proses kompresi dan rekonstruksi pada autoencoder, ilustrasinya disajikan pada Gambar 1.

<img width="940" height="522" alt="image" src="https://github.com/user-attachments/assets/8911c85e-0c6f-40a1-8a67-6261794bc4c6" />

## **2.1.5 _Variational Autoencoder_ (VAE)**
_Variational Autoencoder_ (VAE) merupakan pengembangan dari metode _autoencoder_ tradisional. Pada dasarnya, autoencoder terdiri atas dua komponen utama, yaitu _encoder_ yang bertugas mengubah data berdimensi besar menjadi representasi yang lebih ringkas, serta _decoder_ yang berfungsi mengembalikan representasi tersebut ke bentuk mendekati data awal. Namun, _autoencoder_ biasa cenderung menghasilkan rekonstruksi yang terlalu mirip dengan input sehingga kurang mampu menghasilkan variasi baru. Untuk mengatasi keterbatasan tersebut, VAE memperkenalkan pendekatan probabilistik pada bagian _encoder_ dan menambahkan komponen regularisasi dalam fungsi loss agar ruang laten lebih stabil dan terorganisasi dengan baik (Angelika Septi Rahayu & Santoso, 2023).

Untuk membentuk ruang laten yang terstruktur, VAE membuat _encoder_ menghasilkan parameter distribusi Gaussian, yaitu vektor _mean_ dan standar deviasi. Alih-alih menghasilkan satu titik representasi seperti pada _autoencoder_ biasa, VAE melakukan _sampling_ dari distribusi ini dengan menggunakan _reparameterization trick_ sehingga proses pelatihan tetap dapat dilakukan dengan algoritma berbasis gradien. Fungsi objektif VAE terdiri dari dua bagian, yaitu kesalahan rekonstruksi dan penalti regularisasi berupa nilai _Kullback–Leibler_ (KL) _divergence_. Secara matematis, fungsi loss VAE dinyatakan sebagai:

<img width="523" height="47" alt="image" src="https://github.com/user-attachments/assets/8ed6a0fe-da32-4e71-b9fe-482309956c3f" />

<img width="617" height="157" alt="image" src="https://github.com/user-attachments/assets/c07587b2-2d2a-4e2e-bf48-18e8ce07096d" />

<img width="284" height="44" alt="image" src="https://github.com/user-attachments/assets/d9ea1945-9aef-4209-b70c-c50b731eda56" />

Persamaan (2) menunjukkan bahwa nilai z tidak diambil langsung dari distribusi Gaussian, melainkan diperoleh melalui fungsi transformasi g∅ yang memanfaatkan _noise_ . Dengan cara ini, proses _sampling_ tetap dapat dimasukkan ke dalam alur _backpropagation_. Dengan kombinasi mekanisme rekonstruksi, regularisasi KL, dan _reparameterization trick_, VAE mampu menciptakan ruang laten yang lebih konsisten dan memungkinkan pembangkitan data baru dengan pola yang serupa dengan data asli (Dao et al., 2022).

## **2.1.6 _Kullback-Leibler_ (KL) _Divergence_**
_Kullback-Leibler_ (KL) _Divergence_ merupakan ukuran yang digunakan untuk melihat sejauh mana suatu distribusi probabilitas berbeda dari distribusi acuan. Nilai KL digunakan untuk menilai seberapa besar informasi baru yang terkandung dalam suatu distribusi dibandingkan dengan referensinya. Semakin besar nilai _divergence_, semakin besar pula perbedaan kedua distribusi tersebut. Sebaliknya, nilai yang kecil menunjukkan bahwa distribusi tersebut memiliki kemiripan yang tinggi. Rumus _KL Divergence_ secara umum dapat ditulis sebagai (_KL Divergence_):

<img width="324" height="49" alt="image" src="https://github.com/user-attachments/assets/51117973-db28-4e83-8fae-b903b0a2c029" />

Persamaan (3) dilakukan untuk menghitung selisih logaritmatik antara dua distribusi probabilitas dan menjadi dasar bagi berbagai metode analisis berbasis distribusi. Dalam implementasinya, distribusi biasanya dinormalisasi terlebih dahulu dan nilai yang sangat kecil diberi batas minimum untuk menghindari masalah perhitungan seperti algoritma nol. Teknik ini memastikan proses komputasi tetap stabil dan akurat.

## **2.1.7 _Latent Space Representation_**
Data dengan dimensi tinggi umumnya dapat diproyeksikan ke ruang yang lebih ringkas melalui proses pembentukan latent representations. Representasi laten ini menyimpan informasi penting dari data asli dan sering kali dimanfaatkan untuk berbagai tugas lanjutan. Namun, tanpa adanya mekanisme pengaturan, ruang laten cenderung tidak terstruktur sehingga sulit dikendalikan maupun diinterpretasikan.

Untuk mengatasi masalah tersebut, pendekatan β-VAE digunakan untuk membatasi kapasitas ruang laten sehingga model hanya menangkap fitur penting dari data melalui mekanisme regularisasi pada fungsi loss. Pendekatan ini mendorong model untuk menangkap fitur-fitur paling signifikan dari data sehingga representasi laten menjadi lebih terorganisasi dan lebih mudah dipahami. Representasi yang teratur ini bermanfaat pada berbagai aplikasi generatif, termasuk interpolasi citra (Cristovao et al., 2020).

## **_Residual Block_**
Residual block merupakan elemen arsitektural yang digunakan untuk meningkatkan stabilitas proses pelatihan serta kualitas representasi fitur pada model _Variational Autoencoder_ (VAE). Dalam pendekatan _Multiscale Residual_ VAE, _residual block_ ditempatkan pada bagian _encoder_ maupun _decoder_ untuk menjaga agar informasi penting tetap mengalir dengan baik selama proses propagasi. Mekanisme ini memungkinkan jaringan mempelajari fitur secara lebih mendalam tanpa mengalami kendala umum seperti _vanishing gradient_ ketika jumlah lapisan semakin bertambah.

Selain memberikan jalur informasi tambahan melalui koneksi residual, blok ini juga membantu memperkaya karakteristik ruang laten sehingga representasi yang dihasilkan menjadi lebih halus dan bermakna. Dampaknya terlihat pada peningkatan kualitas rekonstruksi terutama pada citra dengan struktur kompleks. Penerapan residual block telah terbukti efektif dalam berbagai model generatif, termasuk arsitektur VAE multiskala yang memanfaatkan _residual connections_ untuk menghasilkan detail visual yang lebih realistis dan stabil (Diamantis et al., 2024)

## **_Evidance Lower Bound_ (ELBO)**
Pada _Variational Autoencoder_ (VAE), proses pelatihan dilakukan dengan memaksimalkan _Evidence Lower Bound_ (ELBO), yang menjadi batas bawah dari _log-likelihood_ data. Pendekatan ini digunakan karena posterior sebenarnya p∅(z|x) tidak dapat dihitung secara langsung, sehingga diperlukan distribusi pendekatan q∅(z|x) untuk membangun fungsi objektif yang dapat dioptimalkan.
Dalam kerangka _variational inference_, _log-likelihood_ dapat dituliskan ulang sebagaimana ditunjukkan pada Persamaan (4):

<img width="469" height="43" alt="image" src="https://github.com/user-attachments/assets/d8a3dfa0-5c14-4d24-99e3-33fb4479f38e" />

Persamaan (4) memperlihatkan bahwa _log-likelihood_ tersusun dari dua bagian, yaitu nilai ekspektasi terhadap rasio antara model generatif dan distribusi pendekatan, serta _Kullback–Leibler divergence_ antara posterior pendekatan dan posterior sebenarnya. Karena _KL divergence_ selalu bernilai non-negatif, bagian ekspektasi tersebut menjadi batas bawah dari _log-likelihood_ dan disebut sebagai ELBO. Komponen regularisasi dalam ELBO dijelaskan melalui KL divergence antara distribusi pendekatan dan prior, sebagaimana dituliskan pada Persamaan (5):

<img width="462" height="34" alt="image" src="https://github.com/user-attachments/assets/cec88110-3de6-4fce-ae44-d7d3a734f5c5" />

Persamaan (5) ini menggambarkan seberapa jauh distribusi laten hasil _encoder_ menyimpang dari prior. Semakin besar nilainya, semakin besar penalti yang diberikan model untuk menjaga agar ruang laten tetap terstruktur dan tidak menjadi acak. Dengan menggabungkan komponen rekonstruksi dan regularisasi tersebut, batas bawah yang menjadi tujuan optimasi VAE dituliskan pada Persamaan (6):

<img width="466" height="35" alt="image" src="https://github.com/user-attachments/assets/7e26055f-3f5b-4b0c-ad47-f7e1fca6899a" />

Persamaan (6) merangkum dua tujuan penting dalam pelatihan VAE. Bagian pertama (_reconstruction term_) mengukur seberapa baik model dapat membangun kembali input. Bagian kedua (_KL term_) memastikan distribusi laten tetap mendekati prior. Kombinasi keduanya menghasilkan rekonstruksi yang baik sekaligus ruang laten yang stabil dan teratur, yang penting untuk berbagai aplikasi generatif dan analisis representasi.

## **2.1 Kerangka Teori**
Sebagai dasar penyusunan penelitian ini, dilakukan penelusuran terhadap berbagai studi sebelumnya yang berkaitan dengan topik yang disusun. Penelitian-penelitian tersebut dirangkum dalam Tabel 1. berikut:

<img width="456" height="626" alt="image" src="https://github.com/user-attachments/assets/ff40379c-5853-470f-b26d-3522bd48e25e" />

<img width="457" height="601" alt="image" src="https://github.com/user-attachments/assets/9b8d5ea7-1d2b-4c2e-b714-07032cd9efaa" />

Berdasarkan hasil kajian terhadap beberapa penelitian sebelumnya, dapat disimpulkan bahwa penggunaan _Variational Autoencoder_ (VAE) telah diterapkan dalam berbagai konteks, seperti deteksi manipulasi wajah, ekstraksi fitur citra kebakaran, dan prediksi cacat pada mesin. Meskipun demikian, sebagian besar penelitian tersebut lebih menekankan fungsi VAE sebagai alat klasifikasi atau pendukung sistem deteksi, bukan sebagai model generatif yang berfokus pada proses rekonstruksi citra. Selain itu, dataset yang digunakan pada studi terdahulu juga bervariasi dan sebagian besar tidak melibatkan dataset wajah berskala besar seperti CelebA. Di sisi lain, penelitian yang bersifat fundamental seperti studi Kingma dan Welling (2013) memang menjadi dasar konsep VAE, tetapi belum secara khusus mengevaluasi performanya dalam menghasilkan rekonstruksi citra wajah beranotasi kompleks. Dengan demikian, masih terdapat ruang penelitian untuk mengevaluasi kemampuan VAE sebagai model generatif dalam melakukan rekonstruksi citra wajah pada dataset CelebA, sekaligus menguji sejauh mana kualitas hasil rekonstruksi tersebut dapat merepresentasikan karakteristik visual asli. Penelitian ini hadir untuk mengisi celah tersebut melalui implementasi VAE yang difokuskan pada proses rekonstruksi citra wajah sebagai bentuk pendekatan generatif dalam domain _deep learning_.



# **BAB III — METODOLOGI**
## **3.1 Tahapan Kerja**

## **3.2 Dataset**
Dataset yang digunakan dalam penelitian ini adalah CelebFaces Attributes Dataset (CelebA), yaitu kumpulan citra wajah manusia yang sangat populer untuk tugas‑tugas computer vision dan model generatif. Dataset ini berisi 200 gambar wajah dengan resolusi asli 178 × 218 piksel, menampilkan beragam ekspresi, posisi kepala, kondisi pencahayaan, serta variasi karakteristik wajah lainnya. Pada proses pra‑pengolahan di notebook, seluruh gambar diubah ukurannya menjadi 128 × 128 piksel agar sesuai dengan arsitektur model VAE yang digunakan. Dataset CelebA dipilih karena ukurannya yang besar dan keragamannya yang tinggi, sehingga cocok untuk tugas seperti rekonstruksi wajah, pembelajaran distribusi laten wajah, generasi citra baru, dan interpolasi di ruang laten. Transformasi pada data dilakukan menggunakan modul torchvision.transforms, yang meliputi Resize untuk menyesuaikan resolusi gambar, ToTensor untuk mengubah gambar dari format PIL/NumPy menjadi tensor PyTorch berukuran (C, H, W) dengan skala piksel 0–1, serta Normalize dengan mean dan standar deviasi 0.5 agar nilai piksel berada pada rentang –1 hingga 1. Normalisasi ini membantu stabilitas proses pelatihan dan mempercepat konvergensi model VAE.




## **Daftar Pustaka**

Susanto, N., & Pardede, H. (2024). Feature learning using deep variational autoencoder for prediction of defects in car engine (pp. 311–316). https://doi.org/10.1109/ICITRI62858.2024.10699115

Nugroho, H., Susanty, M., Irawan, A., Koyimatu, M., & Yunita, A. (2020). Fully convolutional variational autoencoder for feature extraction of fire detection system. Jurnal Ilmu Komputer dan Informasi, 13(1), 9–15. https://doi.org/10.21609/jiki.v13i1.761

Giger, M., & Csillaghy, A. (2024). Unsupervised anomaly detection with variational autoencoders applied to full-disk solar images. Space Weather, 22, e2023SW003516. https://doi.org/10.1029/2023SW003516

Kingma, D. P., & Welling, M. (2022). Auto-encoding variational Bayes. arXiv. https://arxiv.org/abs/1312.6114

Alfarizi M. Riziq Sirfatullah, Al-farish Muhamad Zidan, Taufiqurrahman Muhamad, Ardiansah Ginan, & Elgar Muhamad. (2023). Penggunaan Python Sebagai Bahasa Pemrograman untuk Machine Learning dan Deep Learning. Karya Ilmiah Mahasiswa Bertauhid (KARIMAH TAUHID), 2(1), 1–6.

Angelika Septi Rahayu, R., & Santoso, H. (2023). Analysis of Fake Face Images: Detecting the Authenticity of Manipulated Images Using Variational Autoencoder Methods and Deep Neural Network Forensics. Sibatik Journal | Volume, 2(9), 2701–2726. https://publish.ojs-indonesia.com/index.php/SIBATIK

César Pérez Curiel. (2022). Análisis del espacio latente en el auto-encoder variacional. https://oa.upm.es/71832/1/TESIS_MASTER_CESAR_PEREZ_CURIEL.pdf

Cristovao, P., Nakada, H., Tanimura, Y., & Asoh, H. (2020). Generating In-Between Images through Learned Latent Space Representation Using Variational Autoencoders. IEEE Access, 8, 149456–149467. https://doi.org/10.1109/ACCESS.2020.3016313

Dao, T. Van, Sato, H., & Kubo, M. (2022). An Attention Mechanism for Combination of CNN and VAE for Image-Based Malware Classification. IEEE Access, 10(August), 85127–85136. https://doi.org/10.1109/ACCESS.2022.3198072

Diamantis, D. E., Gatoula, P., Koulaouzidis, A., & Iakovidis, D. K. (2024). This Intestine Does Not Exist: Multiscale Residual Variational Autoencoder for Realistic Wireless Capsule Endoscopy Image Generation. IEEE Access, 12(February), 25668–25683. https://doi.org/10.1109/ACCESS.2024.3366801

Mauludiah, S. F. (2025). A SYNERGISTIC APPROACH TO E-COMMERCE RECOMMENDER SYSTEM : LOGISTIC REGRESSION AND KULLBACK-LEIBLER DIVERGENCE. http://scioteca.caf.com/bitstream/handle/123456789/1091/RED2017-Eng-8ene.pdf?sequence=12&isAllowed=y%0Ahttp://dx.doi.org/10.1016/j.regsciurbeco.2008.06.005%0Ahttps://www.researchgate.net/publication/305320484_SISTEM_PEMBETUNGAN_TERPUSAT_STRATEGI_MELESTARI

Metlapalli, A. C., Muthusamy, T., & Battula, B. P. (2020). Classification of social media text spam using VAE-CNN and LSTM model. Ingenierie Des Systemes d’Information, 25(6), 747–753. https://doi.org/10.18280/isi.250605

Noguer I Alonso, M. (2024). The Mathematics of Autoencoders and Variational Autoencoders. Ssrn, 1–8. https://www.ssrn.com/abstract=4999896

