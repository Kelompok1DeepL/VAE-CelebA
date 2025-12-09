# LAPORAN  
## PROYEK MATA KULIAH DEEP LEARNING  
### “Implementasi VAE Berbasis Residual Block untuk Rekonstruksi Citra Wajah pada Dataset CelebA”

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
**FAKULTAS ILMU KOMPUTER**  
UNIVERSITAS PEMBANGUNAN NASIONAL "VETERAN" JAWA TIMUR
2025

---

# **BAB I — PENDAHULUAN**

## **Latar Belakang**
<p align="justify">
Dalam bidang kecerdasan buatan (Artificial Intelligence), deep learning merupakan metode yang populer untuk pemrosesan citra, khususnya dengan memanfaatkan kemampuan jaringan saraf tiruan untuk mengekstrak fitur dari data berdimensi tinggi. Hal ini memicu pengembangan model generatif yang bertujuan untuk mempelajari representasi data sehingga sistem mampu menghasilkan sampel baru yang relevan. Salah satu arsitektur generatif yang banyak digunakan adalah **Autoencoder (AE)**.
</p>

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

Secara keseluruhan, CelebA terdiri dari 200 gambar wajah selebriti, dengan total 10.177 identitas berbeda, walaupun informasi nama tidak disertakan. Setiap gambar dilengkapi dengan 40 anotasi atribut biner yang menggambarkan karakteristik wajah tertentu serta lima titik landmark yang meliputi posisi kedua mata, hidung, dan dua titik mulut. Dataset ini juga menyediakan berbagai berkas pendukung, seperti kumpulan gambar wajah yang telah melalui proses cropping dan alignment, pembagian data yang direkomendasikan untuk pelatihan, validasi, pengujian, informasi _bounding box_, serta berkas anotasi atribut untuk seluruh gambar.

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

## **2.1.8 _Residual Block_**
Residual block merupakan elemen arsitektural yang digunakan untuk meningkatkan stabilitas proses pelatihan serta kualitas representasi fitur pada model _Variational Autoencoder_ (VAE). Dalam pendekatan _Multiscale Residual_ VAE, _residual block_ ditempatkan pada bagian _encoder_ maupun _decoder_ untuk menjaga agar informasi penting tetap mengalir dengan baik selama proses propagasi. Mekanisme ini memungkinkan jaringan mempelajari fitur secara lebih mendalam tanpa mengalami kendala umum seperti _vanishing gradient_ ketika jumlah lapisan semakin bertambah.

Selain memberikan jalur informasi tambahan melalui koneksi residual, blok ini juga membantu memperkaya karakteristik ruang laten sehingga representasi yang dihasilkan menjadi lebih halus dan bermakna. Dampaknya terlihat pada peningkatan kualitas rekonstruksi terutama pada citra dengan struktur kompleks. Penerapan residual block telah terbukti efektif dalam berbagai model generatif, termasuk arsitektur VAE multiskala yang memanfaatkan _residual connections_ untuk menghasilkan detail visual yang lebih realistis dan stabil (Diamantis et al., 2024)

## **2.1.9 _Evidance Lower Bound_ (ELBO)**
Pada _Variational Autoencoder_ (VAE), proses pelatihan dilakukan dengan memaksimalkan _Evidence Lower Bound_ (ELBO), yang menjadi batas bawah dari _log-likelihood_ data. Pendekatan ini digunakan karena posterior sebenarnya p∅(z|x) tidak dapat dihitung secara langsung, sehingga diperlukan distribusi pendekatan q∅(z|x) untuk membangun fungsi objektif yang dapat dioptimalkan.
Dalam kerangka _variational inference_, _log-likelihood_ dapat dituliskan ulang sebagaimana ditunjukkan pada Persamaan (4):

<img width="469" height="43" alt="image" src="https://github.com/user-attachments/assets/d8a3dfa0-5c14-4d24-99e3-33fb4479f38e" />

Persamaan (4) memperlihatkan bahwa _log-likelihood_ tersusun dari dua bagian, yaitu nilai ekspektasi terhadap rasio antara model generatif dan distribusi pendekatan, serta _Kullback–Leibler divergence_ antara posterior pendekatan dan posterior sebenarnya. Karena _KL divergence_ selalu bernilai non-negatif, bagian ekspektasi tersebut menjadi batas bawah dari _log-likelihood_ dan disebut sebagai ELBO. Komponen regularisasi dalam ELBO dijelaskan melalui KL divergence antara distribusi pendekatan dan prior, sebagaimana dituliskan pada Persamaan (5):

<img width="429" height="31" alt="image" src="https://github.com/user-attachments/assets/cec8734c-e257-4114-91cb-09da99df6e93" />

Persamaan (5) ini menggambarkan seberapa jauh distribusi laten hasil _encoder_ menyimpang dari prior. Semakin besar nilainya, semakin besar penalti yang diberikan model untuk menjaga agar ruang laten tetap terstruktur dan tidak menjadi acak. Dengan menggabungkan komponen rekonstruksi dan regularisasi tersebut, batas bawah yang menjadi tujuan optimasi VAE dituliskan pada Persamaan (6):

<img width="466" height="35" alt="image" src="https://github.com/user-attachments/assets/7e26055f-3f5b-4b0c-ad47-f7e1fca6899a" />

Persamaan (6) merangkum dua tujuan penting dalam pelatihan VAE. Bagian pertama (_reconstruction term_) mengukur seberapa baik model dapat membangun kembali input. Bagian kedua (_KL term_) memastikan distribusi laten tetap mendekati prior. Kombinasi keduanya menghasilkan rekonstruksi yang baik sekaligus ruang laten yang stabil dan teratur, yang penting untuk berbagai aplikasi generatif dan analisis representasi.

## **2.1 Kerangka Teori**
Sebagai dasar penyusunan penelitian ini, dilakukan penelusuran terhadap berbagai studi sebelumnya yang berkaitan dengan topik yang disusun. Penelitian-penelitian tersebut dirangkum dalam Tabel 1. berikut:

<img width="456" height="626" alt="image" src="https://github.com/user-attachments/assets/ff40379c-5853-470f-b26d-3522bd48e25e" />

<img width="457" height="587" alt="image" src="https://github.com/user-attachments/assets/3b0cba6e-da9e-45d8-ac0f-97130067afec" />


Berdasarkan hasil kajian terhadap beberapa penelitian sebelumnya, dapat disimpulkan bahwa penggunaan _Variational Autoencoder_ (VAE) telah diterapkan dalam berbagai konteks, seperti deteksi manipulasi wajah, ekstraksi fitur citra kebakaran, dan prediksi cacat pada mesin. Meskipun demikian, sebagian besar penelitian tersebut lebih menekankan fungsi VAE sebagai alat klasifikasi atau pendukung sistem deteksi, bukan sebagai model generatif yang berfokus pada proses rekonstruksi citra. Selain itu, dataset yang digunakan pada studi terdahulu juga bervariasi dan sebagian besar tidak melibatkan dataset wajah berskala besar seperti CelebA. Di sisi lain, penelitian yang bersifat fundamental seperti studi Kingma dan Welling (2013) memang menjadi dasar konsep VAE, tetapi belum secara khusus mengevaluasi performanya dalam menghasilkan rekonstruksi citra wajah beranotasi kompleks. Dengan demikian, masih terdapat ruang penelitian untuk mengevaluasi kemampuan VAE sebagai model generatif dalam melakukan rekonstruksi citra wajah pada dataset CelebA, sekaligus menguji sejauh mana kualitas hasil rekonstruksi tersebut dapat merepresentasikan karakteristik visual asli. Penelitian ini hadir untuk mengisi celah tersebut melalui implementasi VAE yang difokuskan pada proses rekonstruksi citra wajah sebagai bentuk pendekatan generatif dalam domain _deep learning_.


# **BAB III — METODOLOGI**
## **3.1 Tahapan Kerja**
<img width="482" height="1091" alt="deeplearning" src="https://github.com/user-attachments/assets/5ebfe791-d4bb-478c-a3d8-8bf466b4d1f6" />

**Mulai Penelitian**
Tahap ini merupakan titik awal penelitian, di mana peneliti menentukan topik, tujuan, dan metode yang akan digunakan. Pada tahap ini juga dirumuskan bahwa penelitian akan menggunakan model Variational Autoencoder (VAE) dengan dataset CelebA untuk melakukan pemodelan dan generasi wajah.

**Ambil Dataset CelebA**
Pada langkah ini, dataset CelebA dikumpulkan dari sumber resmi. Dataset ini berisi ratusan gambar wajah manusia yang digunakan sebagai data latih. Data kemudian disiapkan dalam struktur folder agar mudah diproses oleh sistem.

**Preprocessing Gambar (Resize, ToTensor, Normalization)**
Tahap ini bertujuan untuk menyiapkan data sebelum masuk ke model. Gambar diubah ukurannya agar seragam (misalnya 128×128 piksel), dikonversi menjadi bentuk tensor, dan dinormalisasi supaya nilai piksel berada dalam rentang yang sesuai untuk pembelajaran model. Proses ini penting agar model dapat belajar dengan stabil dan efisien.

**Proses Training (Loss = Reconstruction + KL Divergence)**
Pada tahap ini model dilatih menggunakan data yang sudah diproses. Model VAE mempelajari bagaimana mengompresi gambar ke ruang laten dan merekonstruksinya kembali. Proses pelatihan berfokus pada minimisasi dua komponen loss, yaitu reconstruction loss untuk mengukur kemiripan hasil rekonstruksi dengan gambar asli dan KL divergence untuk mengatur distribusi data di ruang laten.

**Visualisasi**
Setelah proses training, dilakukan visualisasi untuk melihat hasil kerja model. Pada tahap ini ditampilkan grafik loss selama pelatihan dan beberapa contoh hasil keluaran model untuk mempermudah analisis performa.

**Rekonstruksi**
Tahap rekonstruksi menunjukkan kemampuan model dalam membangun kembali gambar yang diberikan sebagai input. Gambar asli dimasukkan ke model, lalu model menghasilkan gambar hasil rekonstruksi. Hasil ini digunakan untuk menilai seberapa baik model memahami struktur data wajah.

**Interpolasi**
Pada tahap ini dilakukan interpolasi di ruang laten, yaitu transisi bertahap antara dua wajah yang berbeda. Tujuannya adalah melihat bagaimana model mempelajari representasi wajah secara kontinu dan bagaimana perubahan antar fitur wajah dapat terjadi secara halus.

**Evaluasi**
Tahap evaluasi dilakukan untuk menilai kinerja model secara keseluruhan. Evaluasi dilakukan dengan membandingkan hasil rekonstruksi, hasil interpolasi, dan nilai loss selama training untuk memastikan model bekerja sesuai tujuan penelitian.

**Kesimpulan Penelitian**
Setelah evaluasi, peneliti menyusun kesimpulan berdasarkan hasil eksperimen. Pada tahap ini dijelaskan apakah model VAE berhasil mempelajari pola wajah dengan baik serta kelebihan dan keterbatasan model yang digunakan.

**Selesai Penelitian**
Tahap ini merupakan akhir dari seluruh rangkaian penelitian. Semua hasil telah dianalisis dan didokumentasikan dalam bentuk laporan atau skripsi, serta diberikan saran untuk pengembangan penelitian selanjutnya.



## **3.2 Dataset**
Dataset yang digunakan dalam penelitian ini adalah CelebFaces Attributes Dataset (CelebA) dari kaggle (https://www.kaggle.com/datasets/jessicali9530/celeba-dataset), yaitu kumpulan citra wajah manusia yang sangat populer untuk tugas‑tugas computer vision dan model generatif. Dataset ini berisi 200 gambar wajah dengan resolusi asli 178 × 218 piksel, menampilkan beragam ekspresi, posisi kepala, kondisi pencahayaan, serta variasi karakteristik wajah lainnya. Pada proses pra‑pengolahan di notebook, seluruh gambar diubah ukurannya menjadi 128 × 128 piksel agar sesuai dengan arsitektur model VAE yang digunakan. Dataset CelebA dipilih karena ukurannya yang besar dan keragamannya yang tinggi, sehingga cocok untuk tugas seperti rekonstruksi wajah, pembelajaran distribusi laten wajah, generasi citra baru, dan interpolasi di ruang laten. Transformasi pada data dilakukan menggunakan modul torchvision.transforms, yang meliputi Resize untuk menyesuaikan resolusi gambar, ToTensor untuk mengubah gambar dari format PIL/NumPy menjadi tensor PyTorch berukuran (C, H, W) dengan skala piksel 0–1, serta Normalize dengan mean dan standar deviasi 0.5 agar nilai piksel berada pada rentang –1 hingga 1. Normalisasi ini membantu stabilitas proses pelatihan dan mempercepat konvergensi model VAE.

# **BAB IV — HASIL & PEMBAHASAN**
## **4.1 Load Dataset**
    
    zip_path = "/content/img_align_celeba.zip"
    extract_path = "/content/celeba"
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)
    
    print("Done!") 

Pada tahap ini, dataset CelebA (CelebFaces Attributes Dataset) dimuat menggunakan ImageFolder. Dataset ini berisi 200 gambar wajah manusia dengan berbagai variasi ekspresi, pencahayaan, dan sudut pandang. Setiap gambar melalui proses preprocessing berupa:
- Resize (128×128 piksel) agar sesuai dengan arsitektur VAE.
- ToTensor() untuk mengubah gambar dari format PIL menjadi tensor PyTorch.
- Normalize(mean=0.5, std=0.5) agar nilai piksel berada pada rentang −1,1 sehingga training lebih stabil.
Dataset kemudian dimasukkan ke DataLoader dengan batch size tertentu (misalnya 64), sehingga gambar dapat diproses secara batch selama training. Tahap ini memastikan model menerima input bersih, seragam, dan siap digunakan.

## **4.2 Build Model (Encoder + Decoder)**

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        return F.relu(x + self.block(x))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 128 → 64
            nn.ReLU(True),
            ResidualBlock(32),

            nn.Conv2d(32, 64, 4, 2, 1), # 64 → 32
            nn.ReLU(True),
            ResidualBlock(64),

            nn.Conv2d(64, 128, 4, 2, 1), # 32 → 16
            nn.ReLU(True),
            ResidualBlock(128),

            nn.Conv2d(128, 256, 4, 2, 1), # 16 → 8
            nn.ReLU(True),
            ResidualBlock(256),

            nn.Conv2d(256, 512, 4, 2, 1), # 8 → 4
            nn.ReLU(True),
            ResidualBlock(512),
        )

        # Detect flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out = self.down(dummy)
            self.flatten_dim = out.numel()

        self.fc_mu = nn.Linear(self.flatten_dim, LATENT_DIM)
        self.fc_logvar = nn.Linear(self.flatten_dim, LATENT_DIM)

    def forward(self, x):
        x = self.down(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    class Decoder(nn.Module):
        def __init__(self, flatten_dim):
            super().__init__()

        self.init_spatial = 4
        self.init_channels = flatten_dim // (self.init_spatial * self.init_spatial)

        self.fc = nn.Linear(LATENT_DIM, self.init_channels * 4 * 4)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.init_channels, 256, 4, 2, 1),
            nn.ReLU(True),
            ResidualBlock(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            ResidualBlock(32),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.init_channels, 4, 4)
        return self.up(x)

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder(self.encoder.flatten_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
        
Pada tahap Build Model, arsitektur VAE disusun dalam bentuk dua komponen utama—Encoder dan Decoder—yang bekerja sama untuk mempelajari representasi laten dari gambar wajah. Encoder mengambil gambar berukuran 128×128×3 dan melakukan proses ekstraksi fitur melalui beberapa lapisan Convolutional dan ReLU yang disertai ResidualBlock agar pembelajaran fitur lebih stabil. Proses ini sekaligus melakukan downsampling bertahap hingga menghasilkan feature map berukuran 4×4 dengan channel besar, kemudian di-flatten untuk menentukan ukuran vektor yang akan dihubungkan ke dua fully connected layer. Dua layer ini membentuk output berupa μ (mean) dan logσ² (log-variance), yang merupakan representasi statistik dari distribusi laten. Selanjutnya, VAE menggunakan reparameterization trick untuk mengubah kedua nilai tersebut menjadi vektor laten z yang dapat di-backpropagate. Pada sisi lain, Decoder mengambil vektor laten z dan mengubahnya kembali menjadi gambar melalui fully connected layer yang membentuk kembali tensor 4×4×channel, kemudian memprosesnya melalui beberapa lapisan ConvTranspose2D dengan ResidualBlock untuk melakukan upsampling hingga ukuran kembali menjadi 128×128×3. Proses ini memungkinkan model melakuk

## **4.3 Training Loop**

    loss_history = []
    vae.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(DEVICE)

        recon, mu, logvar = vae(images)
        loss = vae_loss(recon, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Batch loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataset)
    print(f"=== Epoch {epoch}/{EPOCHS} finished. Avg loss per image: {avg_loss:.4f} ===")

    # SIMPAN LOSS
    loss_history.append(avg_loss)

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_loss': avg_loss
    }, CHECKPOINT_PATH)

    print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    # Generate samples
    vae.eval()
    with torch.no_grad():
        z = torch.randn(16, LATENT_DIM).to(DEVICE)
        samples = vae.decoder(z)
        sample_path = os.path.join(SAMPLES_DIR, f"samples_epoch_{epoch}.png")
        save_reconstructed_grid(samples.cpu(), sample_path, nrow=4)

    vae.train()

    
Pada tahap training, proses dimulai dengan forward pass, yaitu gambar input dimasukkan ke dalam encoder untuk menghasilkan dua parameter, yaitu mu dan logvar, yang merepresentasikan distribusi laten. Dari parameter ini, model melakukan proses reparameterization untuk menghasilkan nilai laten z, yang kemudian diteruskan ke decoder untuk menghasilkan citra rekonstruksi. Setelah rekonstruksi dihasilkan, model menghitung nilai loss yang terdiri dari dua komponen: Reconstruction Loss (MSE) yang mengukur seberapa mirip citra hasil rekonstruksi dengan citra asli, serta KL Divergence yang memastikan bahwa distribusi laten mendekati distribusi Gaussian standar. Selanjutnya dilakukan backward pass, yaitu proses propagasi balik menggunakan optimizer.zero_grad(), loss.backward(), dan optimizer.step() untuk memperbarui bobot model berdasarkan error yang diperoleh. Pada setiap epoch, model juga menyimpan checkpoint agar hasil pelatihan dapat dipantau dan dilanjutkan, serta menghasilkan sampel wajah baru dari ruang laten. Seluruh proses ini diulang selama beberapa epoch, dan pada tiap epoch dicatat nilai loss rata-rata untuk melihat perkembangan performa model selama pelatihan.

<img width="485" height="421" alt="image" src="https://github.com/user-attachments/assets/c90337dc-0145-4b3a-bb6b-d312800eb9c4" />


Selama proses training selama 100 epoch, nilai loss model VAE menunjukkan penurunan yang konsisten dari sekitar 288 pada epoch awal hingga mencapai sekitar 221 pada epoch terakhir. Pola penurunan ini terlihat stabil meskipun terdapat sedikit fluktuasi antar‑batch, yang merupakan hal wajar pada training model generatif. Penurunan loss ini mencerminkan bahwa model semakin baik dalam melakukan dua hal utama: merekonstruksi gambar input dan membentuk distribusi latent yang sesuai dengan prior Gaussian. Setiap epoch menghasilkan checkpoint dan sampel gambar baru, yang memperlihatkan bahwa kualitas rekonstruksi dan gambar hasil generasi semakin meningkat seiring menurunnya loss. Secara keseluruhan, hasil training loop menunjukkan bahwa VAE berhasil belajar dan mengalami konvergensi yang baik.

## **4.4 Train Loss**
<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/fd42ad8f-bb74-42be-8a57-60430433c9b3" />
Grafik Training Loss per Epoch menunjukkan bagaimana nilai error pada model Variational Autoencoder (VAE) berubah sepanjang proses pelatihan dari epoch 1 hingga 100. Pada awal pelatihan, nilai loss berada pada kisaran yang cukup tinggi, sekitar 290, dan mengalami fluktuasi yang cukup tajam pada 10–15 epoch pertama. Hal ini merupakan kondisi wajar karena model masih berada pada tahap awal pembelajaran, bobot masih acak, dan proses reparameterization (yang melibatkan mu dan logvar) belum stabil sepenuhnya. Setelah melewati fase awal yang fluktuatif, grafik menunjukkan penurunan loss yang lebih konsisten dan bertahap, menandakan bahwa model mulai berhasil mempelajari pola wajah dalam dataset dan meningkatkan kemampuan rekonstruksi. Memasuki epoch 40 hingga 80, nilai loss terus menurun secara stabil dengan perubahan yang lebih halus, menunjukkan bahwa model berada dalam fase optimasi yang baik. Pada epoch mendekati 100, grafik tampak mulai mendatar, menandakan bahwa model mendekati kondisi konvergensi, di mana penurunan loss tidak lagi signifikan meskipun pelatihan dilanjutkan. Secara keseluruhan, grafik ini menunjukkan bahwa proses pelatihan berjalan dengan baik dan stabil, di mana nilai loss menurun secara signifikan dari awal hingga akhir epoch, yang berarti model berhasil belajar representasi laten dan mampu melakukan rekonstruksi gambar dengan semakin baik seiring bertambahnya epoch.

## **4.5 Hasil Training: Rekonstruksi**
<img width="1244" height="350" alt="image" src="https://github.com/user-attachments/assets/1e7cf9c3-3016-4b1e-9e0f-5cce71ae544e" />
Pada visualisasi rekonstruksi, baris pertama menampilkan gambar asli yang diambil dari dataset CelebA, sedangkan baris kedua menunjukkan hasil rekonstruksi yang dihasilkan oleh decoder setelah melalui proses encoding dan sampling latent. Dari gambar terlihat bahwa model mampu mempertahankan struktur global wajah, seperti bentuk muka, posisi mata, kontur hidung, serta warna rambut. Setiap wajah pada baris kedua masih menyerupai wajah pada baris pertama dalam hal komposisi dan proporsi. Namun, rekonstruksi terlihat lebih buram dan kurang detail, terutama pada area rambut dan tekstur kulit. Beberapa hasil juga tampak mengalami sedikit distorsi atau noise pada bagian pinggir gambar, yang menunjukkan bahwa model masih memiliki keterbatasan dalam menghasilkan detail resolusi tinggi. Meski demikian, kecocokan bentuk dan fitur utama menunjukkan bahwa VAE dengan residual blocks telah berhasil mempelajari pola distribusi wajah secara memadai sehingga dapat mengembalikan citra dengan konsistensi yang baik pada tingkat global.

## **4.6 Hasil: Latent Interpolation**
<img width="1570" height="199" alt="image" src="https://github.com/user-attachments/assets/4d75dab7-425b-4eba-aa96-77ca8bac5f1c" />
Pada bagian latent interpolation, dua wajah yang berbeda dijadikan titik awal dan titik akhir. Gambar interpolasi di antara keduanya menunjukkan perubahan bertahap dari wajah A menuju wajah B. Perubahan terjadi secara mulus: mulai dari bentuk wajah, ekspresi, tekstur rambut, hingga tone warna kulit. Pada gambar yang kamu tampilkan, wajah pada sisi kiri terlihat lebih gelap dan memiliki gaya rambut tertentu, kemudian secara perlahan berubah menjadi wajah dengan rambut lebih terang dan bentuk wajah berbeda pada sisi kanan. Transisi antar frame terlihat halus dan konsisten, tanpa perubahan mendadak atau artefak besar, yang menunjukkan bahwa latent space model tersusun dengan baik dan bersifat kontinu. Ini membuktikan bahwa VAE tidak sekadar menghafal gambar, tetapi benar‑benar mempelajari representasi abstrak yang memungkinkan perpindahan fitur secara logis di ruang laten.

# **BAB V — KESIMPULAN**
Penelitian ini berhasil menunjukkan bahwa Variational Autoencoder (VAE) dengan tambahan Residual Blocks mampu mempelajari representasi laten dari citra wajah CelebA dan menghasilkan rekonstruksi serta interpolasi yang cukup baik meskipun dataset yang digunakan relatif kecil. Model yang dibangun mampu memampatkan citra wajah ke dalam ruang laten berdimensi rendah dan mengembalikannya menjadi citra yang menyerupai input. Hasil rekonstruksi menunjukkan bahwa struktur global wajah seperti bentuk muka, posisi mata, hidung, dan rambut dapat dipertahankan, meskipun detail halus seperti tekstur kulit masih tampak buram.

Selama proses training, loss model menurun secara konsisten dari sekitar 288 menjadi sekitar 221, menandakan bahwa model berhasil belajar dan mencapai konvergensi. Penggunaan Residual Blocks terbukti membantu stabilitas proses pelatihan dan mengurangi hilangnya informasi pada tahap encoding maupun decoding. Selain itu, hasil interpolasi pada ruang laten memperlihatkan transisi wajah yang halus dan kontinu dari satu identitas ke identitas lain. Hal ini membuktikan bahwa model tidak hanya menghafal gambar, tetapi juga mempelajari struktur laten yang bermakna dan terorganisasi. Secara keseluruhan, penelitian ini menunjukkan bahwa VAE dengan Residual Blocks dapat bekerja efektif pada citra wajah dan mampu menghasilkan rekonstruksi serta citra sintetis yang cukup realistis.

# **DAFTAR PUSTAKA**
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

