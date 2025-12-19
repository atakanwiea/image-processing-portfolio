# Görüntü İşleme Projesi 

Kısa açıklama: Bu repoda bir görüntü işleme ödevi/projesi bulunuyor. Çalışma akademik kapsamlıdır ve ödev kapsamında istenen adımlar uygulandı.

Önerilen repository adı: `image-processing-portfolio` (veya `goruntu_isleme_rapor`).

İçerik (ödev kapsamı):

2.1. Ön İşleme ve Sayısallaştırma
- Kullanılan görüntülerin temel özellikleri incelenmesi (boyut, veri tipi, dinamik aralık).
- Gerektiğinde yeniden örnekleme yöntemleri kullanılarak ortak boyuta getirilmesi.

2.2. Histogram Tabanlı İşlemler
- Gri seviye dönüşümü ve histogram analizleri.
- Parlaklık ve kontrast düzenlemeleri.
- Farklı gamma değerlerinin etkilerinin incelenmesi.
- En az bir görüntüde histogram eşitlemenin uygulanması ve etkilerinin tartışılması.

2.3. Gürültü Modellemesi ve Gürültü Azaltma
- En az iki görüntüye Gauss ve tuz-biber gürültüsü eklenmesi.
- Ortalama, Gaussian ve median filtrelerin uygulanması.
- Sonuçların PSNR ve MSE gibi metriklerle değerlendirilmesi.

2.4. Uzamsal Filtreleme ve Keskinleştirme
- Laplacian tabanlı keskinleştirme, unsharp masking ve high-boost filtreleme.
- Parametre değişimlerinin görüntü niteliği üzerindeki etkilerinin incelenmesi.

2.5. Kenar Tespiti
- Sobel, Prewitt ve Canny yöntemlerinin uygulanması.
- Yöntemler arasındaki farkların gürültü dayanıklılığı ve kenar doğruluğu bakımından analizi.

2.6. Renk Uzayı Dönüşümleri
- RGB görüntülerinin HSV ve YCbCr renk uzaylarına dönüştürülmesi.
- V veya Y kanalı üzerinde parlaklık düzeltmeleri yapılarak tekrar RGB’ye dönüştürülmesi.

2.7. Özellik Çıkarımı ve Anahtar Nokta Analizi
- Köşe tespitinin uygulanması.
- Tanımlayıcı ile anahtar nokta ve öznitelik çıkarımı.
- Parametre değişimlerinin eşleşme başarımı üzerindeki etkilerinin değerlendirilmesi.

2.8. Özellik Eşleştirme ve Panorama Oluşturma
- Descriptor tabanlı eşleştirme.
- Yanlış eşleşmelerin RANSAC ile elenmesi.
- Homografi tahmini ile görüntülerin aynı düzleme projeksiyonu.
- Panorama oluşturma ve geçiş bölgelerinin uyumlu bir şekilde birleştirilmesi.

Kullanılan dosyalar ve klasör yapısı örnek:
- `step_2_1_preprocessing.py`, `step_2_2_histogram.py`, ..., `step_2_8_panorama.py`
- `images/` — örnek görüntüler
- `outputs/` — çıktı görüntüler ve metrikler

Hızlı kullanım:
1. Ortamı hazırlayın (örnek `venv` veya conda environment).
2. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

3. İstediğiniz adımı çalıştırın, örn:

```bash
python step_2_2_histogram.py
```

İletişim: GitHub: https://github.com/atakanwiea

Lisans: MIT
