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
Dataset CelebFaces Attributes (CelebA) merupakan salah satu dataset wajah yang banyak digunakan dalam bidang computer vision dan deep learning, terutama untuk berbagai aplikasi yang membutuhkan identifikasi wajah dan analisis atribut wajah. Dataset dirancang untuk mendukung beragam penelitian, mulai dari pengenalan ekspresi, pendeteksian atribut seperti seseorang yang tersenyum, memiliki rambut berwarna tertentu, hingga penggunaan kacamata. Gambar-gambar pada CelebA mencakup variasi pose, kondisi latar belakang yang bervariasi, serta individu dari berbagai karakteristik, sehingga sangat cocok untuk proses pelatihan dan pengujian model berbasis citra wajah. Dataset ini awalnya dikembangkan oleh tim penelitian di MMLAB, The Chinese University of Hong Kong.

Secara keseluruhan, CelebA terdiri dari 202.599 gambar wajah selebriti, dengan total 10.177 identitas berbeda, walaupun informasi nama tidak disertakan. Setiap gambar dilengkapi dengan 40 anotasi atribut biner yang menggambarkan karakteristik wajah tertentu serta lima titik landmark yang meliputi posisi kedua mata, hidung, dan dua titik mulut. Dataset ini juga menyediakan berbagai berkas pendukung, seperti kumpulan gambar wajah yang telah melalui proses cropping dan alignment, pembagian data yang direkomendasikan untuk pelatihan, validasi, pengujian, informasi bounding box, serta berkas anotasi atribut untuk seluruh gambar.

CelebA digunakan secara luas untuk keperluan penelitian akademik dan tersedia hanya untuk penggunaan non-komersial. Dataset ini telah menjadi dasar bagi banyak studi terkait deteksi dan pengenalan wajah, termasuk penelitian oleh Yang, Luo, Loy, dan Tang (2015) yang mengembangkan pendekatan deteksi wajah berbasis pembelajaran mendalam. Dengan jumlah data yang besar, anotasi yang lengkap, serta keberagaman gambar yang tinggi, CelebA menjadi pilihan ideal untuk membangun dan mengevaluasi model-model yang bertujuan mengenali atribut wajah atau menghasilkan kembali citra wajah secara otomatis.

## **2.1.2 _Deep Learning_**
Deep Learning merupakan bagian dari Machine Learning yang dikembangkan berdasarkan cara kerja jaringan saraf biologis pada otak manusia. Pendekatan ini menggunakan model yang disebut Jaringan Saraf Tiruan (Artificial Neural Networks/ANN), yang tersusun dari lapisan-lapisan neuron buatan untuk memproses informasi secara berjenjang. Di dalam Deep Learning, terdapat berbagai jenis arsitektur yang dirancang untuk tugas tertentu, seperti Convolutional Neural Network (CNN) untuk pengolahan citra, Recurrent Neural Network (RNN) dan Long Short-Term Memory (LSTM) untuk data berurutan, serta Self Organizing Maps (SOM) untuk pemetaan dan pengelompokan data (Sitasi Deep Learning).

## **2.1.3 _Convolutional Neural Network_ (CNN)**
Convolutional Neural Network (CNN) adalah arsitektur deep learning yang dirancang untuk mempelajari pola penting dari data yang memiliki struktur, seperti citra maupun teks. Model ini tersusun atas beberapa jenis lapisan yang bekerja secara bertahap. Bagian konvolusi berfungsi mengekstraksi ciri menggunakan kernel yang bergerak melintasi data input dan menghasilkan feature map yang mewakili pola-pola penting. Setelah itu, lapisan pooling mereduksi ukuran representasi tersebut sehingga model menjadi lebih efisien dan lebih tahan terhadap perubahan posisi atau pergeseran fitur.

Hasil ekstraksi fitur kemudian diratakan (flattening) dan diteruskan ke lapisan fully connected untuk proses klasifikasi akhir. Keunggulan utama CNN terletak pada kemampuannya melakukan ekstraksi fitur secara otomatis tanpa memerlukan rekayasa fitur manual, serta sifat spatial invariance yang membuat model tetap mampu mengenali pola meskipun terjadi perubahan posisi atau bentuk kecil pada input. Pendekatan ini menjadikan CNN efektif digunakan dalam berbagai tugas klasifikasi berbasis gambar maupun data teks berurutan (Classification of Social Media).

## **2.1.4 _Autoencoder_**
Autoencoder merupakan jaringan saraf yang dirancang untuk mempelajari cara merekonstruksi kembali data masukan. Model ini terdiri dari encoder yang memampatkan input menjadi representasi berdimensi rendah, serta decoder yang menghasilkan rekonstruksi dari representasi tersebut. Meskipun mampu menyalin ulang data, nilai utama autoencoder sering terletak pada representasi latennya yang dapat digunakan untuk berbagai tugas analisis (TESIS_MASTER). Untuk memberikan gambaran visual mengenai proses kompresi dan rekonstruksi pada autoencoder, ilustrasinya disajikan pada Gambar 1.


## **2.1.5 _Variational Autoencoder_ (VAE)**
Variational Autoencoder (VAE) merupakan pengembangan dari metode autoencoder tradisional. Pada dasarnya, autoencoder terdiri atas dua komponen utama, yaitu encoder yang bertugas mengubah data berdimensi besar menjadi representasi yang lebih ringkas, serta decoder yang berfungsi mengembalikan representasi tersebut ke bentuk mendekati data awal. Namun, autoencoder biasa cenderung menghasilkan rekonstruksi yang terlalu mirip dengan input sehingga kurang mampu menghasilkan variasi baru. Untuk mengatasi keterbatasan tersebut, VAE memperkenalkan pendekatan probabilistik pada bagian encoder dan menambahkan komponen regularisasi dalam fungsi loss agar ruang laten lebih stabil dan terorganisasi dengan baik (Sitasi VAE).

Untuk membentuk ruang laten yang terstruktur, VAE membuat encoder menghasilkan parameter distribusi Gaussian, yaitu vektor mean dan standar deviasi. Alih-alih menghasilkan satu titik representasi seperti pada autoencoder biasa, VAE melakukan sampling dari distribusi ini dengan menggunakan reparameterization trick sehingga proses pelatihan tetap dapat dilakukan dengan algoritma berbasis gradien. Fungsi objektif VAE terdiri dari dua bagian, yaitu kesalahan rekonstruksi dan penalti regularisasi berupa nilai Kullback–Leibler (KL) divergence. Secara matematis, fungsi loss VAE dinyatakan sebagai:

<img width="523" height="47" alt="image" src="https://github.com/user-attachments/assets/8ed6a0fe-da32-4e71-b9fe-482309956c3f" />

<img width="617" height="157" alt="image" src="https://github.com/user-attachments/assets/c07587b2-2d2a-4e2e-bf48-18e8ce07096d" />

<img width="284" height="44" alt="image" src="https://github.com/user-attachments/assets/d9ea1945-9aef-4209-b70c-c50b731eda56" />

Persamaan (2) menunjukkan bahwa nilai z tidak diambil langsung dari distribusi Gaussian, melainkan diperoleh melalui fungsi transformasi g∅ yang memanfaatkan noise . Dengan cara ini, proses sampling tetap dapat dimasukkan ke dalam alur backpropagation. Dengan kombinasi mekanisme rekonstruksi, regularisasi KL, dan reparameterization trick, VAE mampu menciptakan ruang laten yang lebih konsisten dan memungkinkan pembangkitan data baru dengan pola yang serupa dengan data asli (sitasi VAE 2).

## **2.1.6 _Kullback-Leibler_ (KL) _Divergence_**
Kullback-Leibler (KL) Divergence merupakan ukuran yang digunakan untuk melihat sejauh mana suatu distribusi probabilitas berbeda dari distribusi acuan. Nilai KL digunakan untuk menilai seberapa besar informasi baru yang terkandung dalam suatu distribusi dibandingkan dengan referensinya. Semakin besar nilai divergence, semakin besar pula perbedaan kedua distribusi tersebut. Sebaliknya, nilai yang kecil menunjukkan bahwa distribusi tersebut memiliki kemiripan yang tinggi. Rumus KL Divergence secara umum dapat ditulis sebagai (KL Divergence):

<img width="324" height="49" alt="image" src="https://github.com/user-attachments/assets/51117973-db28-4e83-8fae-b903b0a2c029" />

Persamaan (3) dilakukan untuk menghitung selisih logaritmatik antara dua distribusi probabilitas dan menjadi dasar bagi berbagai metode analisis berbasis distribusi. Dalam implementasinya, distribusi biasanya dinormalisasi terlebih dahulu dan nilai yang sangat kecil diberi batas minimum untuk menghindari masalah perhitungan seperti algoritma nol. Teknik ini memastikan proses komputasi tetap stabil dan akurat.

## **2.1.7 _Latent Space Representation_**
Data dengan dimensi tinggi umumnya dapat diproyeksikan ke ruang yang lebih ringkas melalui proses pembentukan latent representations. Representasi laten ini menyimpan informasi penting dari data asli dan sering kali dimanfaatkan untuk berbagai tugas lanjutan. Namun, tanpa adanya mekanisme pengaturan, ruang laten cenderung tidak terstruktur sehingga sulit dikendalikan maupun diinterpretasikan.

Untuk mengatasi masalah tersebut, pendekatan β-VAE digunakan untuk membatasi kapasitas ruang laten sehingga model hanya menangkap fitur penting dari data melalui mekanisme regularisasi pada fungsi loss. Pendekatan ini mendorong model untuk menangkap fitur-fitur paling signifikan dari data sehingga representasi laten menjadi lebih terorganisasi dan lebih mudah dipahami. Representasi yang teratur ini bermanfaat pada berbagai aplikasi generatif, termasuk interpolasi citra. (Latent Representation)

## **_Evidance Lower Bound_ (ELBO)**
Pada Variational Autoencoder (VAE), proses pelatihan dilakukan dengan memaksimalkan Evidence Lower Bound (ELBO), yang menjadi batas bawah dari log-likelihood data. Pendekatan ini digunakan karena posterior sebenarnya p∅(z|x) tidak dapat dihitung secara langsung, sehingga diperlukan distribusi pendekatan q∅(z|x) untuk membangun fungsi objektif yang dapat dioptimalkan.
Dalam kerangka variational inference, log-likelihood dapat dituliskan ulang sebagaimana ditunjukkan pada Persamaan (4):


Persamaan (4) memperlihatkan bahwa log-likelihood tersusun dari dua bagian, yaitu nilai ekspektasi terhadap rasio antara model generatif dan distribusi pendekatan, serta Kullback–Leibler divergence antara posterior pendekatan dan posterior sebenarnya. Karena KL divergence selalu bernilai non-negatif, bagian ekspektasi tersebut menjadi batas bawah dari log-likelihood dan disebut sebagai ELBO. Komponen regularisasi dalam ELBO dijelaskan melalui KL divergence antara distribusi pendekatan dan prior, sebagaimana dituliskan pada Persamaan (5):


Persamaan (5) ini menggambarkan seberapa jauh distribusi laten hasil encoder menyimpang dari prior. Semakin besar nilainya, semakin besar penalti yang diberikan model untuk menjaga agar ruang laten tetap terstruktur dan tidak menjadi acak. Dengan menggabungkan komponen rekonstruksi dan regularisasi tersebut, batas bawah yang menjadi tujuan optimasi VAE dituliskan pada Persamaan (6):


Persamaan (6) merangkum dua tujuan penting dalam pelatihan VAE. Bagian pertama (reconstruction term) mengukur seberapa baik model dapat membangun kembali input. Bagian kedua (KL term) memastikan distribusi laten tetap mendekati prior. Kombinasi keduanya menghasilkan rekonstruksi yang baik sekaligus ruang laten yang stabil dan teratur, yang penting untuk berbagai aplikasi generatif dan analisis representasi.

