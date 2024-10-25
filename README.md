# XOR Problemi İçin Yapay Sinir Ağı

Bu proje, yapay sinir ağı kullanarak XOR problemini çözmek amacıyla geliştirilmiştir. Python ve `numpy` kütüphanesi ile sıfırdan oluşturulmuş bu çok katmanlı sinir ağı modeli (MLP), XOR probleminin öğrenme sürecini, ağırlıkların güncellenmesini ve eğitim sonrası ağırlık görselleştirmesini içerir. Bu, derin öğrenme algoritmalarının temel prensiplerini anlamak için etkili bir örnektir.

## Özellikler
- **Çok Katmanlı Sinir Ağı (MLP)**: XOR problemine çözüm bulmak için ileri beslemeli bir ağ kullanır.
- **Ağırlık ve Bias Görselleştirme**: Modelin başlangıç ve eğitim sonrası ağırlıkları ve bias değerleri `networkx` kütüphanesi ile görselleştirilir.
- **Sigmoid Aktivasyon Fonksiyonu**: Hem ileri hem de geri yayılımda sigmoid aktivasyon fonksiyonu kullanılarak öğrenme süreci optimize edilir.
- **Gizli Katman Nöron Sayısı Ayarı**: Kullanıcı, gizli katmandaki nöron sayısını giriş sırasında belirleyebilir.

## Kurulum

Bu projeyi kullanmak için aşağıdaki kütüphaneleri kurmanız gerekmektedir:

```bash
pip install numpy matplotlib networkx
Program başladığında sizden gizli katmandaki nöron sayısını girmenizi isteyecektir. Nöron sayısını girdikten sonra ağ eğitilecek ve aşağıdaki çıktılar gösterilecektir:

Başlangıç ağırlıkları ve biaslar
Eğitim sürecindeki hata grafiği
Eğitim sonrası ağırlık ve biaslar ile güncellenmiş sinir ağı yapısı görselleştirilir.
Kodun Detaylı Açıklaması
Sigmoid Fonksiyonu ve Türevi: sigmoid ve sigmoid_derivative fonksiyonları, aktivasyon fonksiyonu olarak kullanılır. Sigmoid, giriş verilerini 0 ve 1 arasında normalize eder ve türevi, geri yayılımda hata güncellemeleri için gereklidir.

NeuralNetwork Sınıfı: Ana yapay sinir ağı sınıfıdır. İçeriği:

__init__: Modeli başlatır ve ağırlıkları rastgele değerlerle doldurur.
forward: Girişten çıkışa kadar ileri besleme sürecini yönetir.
backward: Hata geri yayılımı ile ağırlık güncellemesini gerçekleştirir.
train: Modeli belirli bir epoch sayısı boyunca eğitir.
visualize_network: Ağ yapısını ve ağırlıkları networkx ile görselleştirir.
Eğitim Süreci: Kullanıcıdan alınan gizli katman nöron sayısına göre sinir ağı yapılandırılır ve 10.000 epoch boyunca eğitilir. Her 1000 epoch'ta eğitim hatası konsola yazdırılır.

Çıktılar ve Görselleştirme
Ağırlık Görselleştirme: Eğitim öncesi ve sonrası ağırlık ve biaslar, ağ yapısı üzerinde gösterilir.
Eğitim Hatası Grafiği: Her epoch'ta hata değerleri kaydedilir ve grafik üzerinde çizilir.
Katkıda Bulunma
Bu projeye katkıda bulunmak isterseniz bir pull request açabilir ya da bir issue ile önerilerinizi sunabilirsiniz. Kodunuzu projeye katkı kurallarına göre düzenlediğinizden emin olun.
