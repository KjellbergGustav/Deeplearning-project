# Results

## DnCNN results
The DnCnn was run 6 times and the average loss (L2 - MSE) was calculated:

```
Averge loss over 6 runs with batch sizes:  [32, 64, 128, 256, 512, 1024]  and a data set size of (10000, 28, 28, 1)
Gauss:  0.015856646622220676
Speckle:  0.007151896366849542
Salt and Pepper:  0.008568066637963057
Block:  0.076985248674949
Border:  0.003484490909613669
No noise:  0.003379359724931419
Joint average:  0.019237618156087894
```
### The runs
```
[INFO] loading MNIST dataset...
Loading existing model...
2020-05-10 13:39:48.121469: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 13:39:48.139714: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f8b937a0330 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 13:39:48.139737: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Evaluate model on Gaussian noise:
313/313 [==============================] - 15s 49ms/step - loss: 0.0159
Evaluate model on Speckle noise:
313/313 [==============================] - 15s 47ms/step - loss: 0.0072
Evaluate model on Salt&Pepper noise:
313/313 [==============================] - 15s 46ms/step - loss: 0.0086
Evaluate model on Block noise:
313/313 [==============================] - 14s 46ms/step - loss: 0.0766
Evaluate model on Border noise:
313/313 [==============================] - 15s 47ms/step - loss: 0.0035
Evaluate model on no noise:
313/313 [==============================] - 15s 48ms/step - loss: 0.0034
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.01587432250380516   0.007153005339205265   0.008554251864552498   0.0766226053237915   0.0034867404028773308   0.003381303744390607
Evaluate model on Gaussian noise:
157/157 [==============================] - 13s 85ms/step - loss: 0.0159
Evaluate model on Speckle noise:
157/157 [==============================] - 14s 86ms/step - loss: 0.0072
Evaluate model on Salt&Pepper noise:
157/157 [==============================] - 13s 85ms/step - loss: 0.0086
Evaluate model on Block noise:
157/157 [==============================] - 14s 86ms/step - loss: 0.0771
Evaluate model on Border noise:
157/157 [==============================] - 13s 86ms/step - loss: 0.0035
Evaluate model on no noise:
157/157 [==============================] - 14s 89ms/step - loss: 0.0034
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.015897151082754135   0.007153404876589775   0.008561266586184502   0.07712455093860626   0.0034852195531129837   0.0033801281824707985
Evaluate model on Gaussian noise:
79/79 [==============================] - 12s 152ms/step - loss: 0.0158
Evaluate model on Speckle noise:
79/79 [==============================] - 12s 154ms/step - loss: 0.0071
Evaluate model on Salt&Pepper noise:
79/79 [==============================] - 12s 151ms/step - loss: 0.0086
Evaluate model on Block noise:
79/79 [==============================] - 12s 152ms/step - loss: 0.0769
Evaluate model on Border noise:
79/79 [==============================] - 12s 152ms/step - loss: 0.0035
Evaluate model on no noise:
79/79 [==============================] - 12s 151ms/step - loss: 0.0034
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.015768107026815414   0.0071353791281580925   0.008587158285081387   0.07693199813365936   0.0034822067245841026   0.003377796383574605
Evaluate model on Gaussian noise:
40/40 [==============================] - 10s 262ms/step - loss: 0.0158
Evaluate model on Speckle noise:
40/40 [==============================] - 10s 252ms/step - loss: 0.0072
Evaluate model on Salt&Pepper noise:
40/40 [==============================] - 11s 271ms/step - loss: 0.0086
Evaluate model on Block noise:
40/40 [==============================] - 11s 272ms/step - loss: 0.0766
Evaluate model on Border noise:
40/40 [==============================] - 11s 272ms/step - loss: 0.0035
Evaluate model on no noise:
40/40 [==============================] - 11s 272ms/step - loss: 0.0034
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.015798885375261307   0.007155175320804119   0.008571786805987358   0.07660502195358276   0.003476293757557869   0.003373220097273588
Evaluate model on Gaussian noise:
20/20 [==============================] - 10s 495ms/step - loss: 0.0159
Evaluate model on Speckle noise:
20/20 [==============================] - 10s 498ms/step - loss: 0.0071
Evaluate model on Salt&Pepper noise:
20/20 [==============================] - 10s 496ms/step - loss: 0.0086
Evaluate model on Block noise:
20/20 [==============================] - 10s 498ms/step - loss: 0.0772
Evaluate model on Border noise:
20/20 [==============================] - 10s 494ms/step - loss: 0.0035
Evaluate model on no noise:
20/20 [==============================] - 10s 498ms/step - loss: 0.0034
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.01594356633722782   0.007148349191993475   0.008580734021961689   0.07722321152687073   0.003484180662781   0.0033777158241719007
Evaluate model on Gaussian noise:
10/10 [==============================] - 9s 900ms/step - loss: 0.0159
Evaluate model on Speckle noise:
10/10 [==============================] - 9s 898ms/step - loss: 0.0072
Evaluate model on Salt&Pepper noise:
10/10 [==============================] - 9s 907ms/step - loss: 0.0086
Evaluate model on Block noise:
10/10 [==============================] - 9s 898ms/step - loss: 0.0774
Evaluate model on Border noise:
10/10 [==============================] - 9s 858ms/step - loss: 0.0035
Evaluate model on no noise:
10/10 [==============================] - 9s 889ms/step - loss: 0.0034
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.015857847407460213   0.007166064344346523   0.008553202264010906   0.07740410417318344   0.0034923043567687273   0.003385994117707014
```

## AE trained on all noises

```
Averge loss over 6 runs with batch sizes:  [32, 64, 128, 256, 512, 1024]  and a data set size of (10000, 28, 28, 1)
Gauss:  0.018516422249376774
Speckle:  0.01113004811728994
Salt and Pepper:  0.012291195647170147
Block:  0.08078224460283916
Border:  0.008464179777850708
No noise:  0.00827620162939032
Joint average:  0.023243382003986176
```

### The runs

```
Evaluate model on Gaussian noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0185
Evaluate model on Speckle noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0111
Evaluate model on Salt&Pepper noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0123
Evaluate model on Block noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0805
Evaluate model on Border noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0085
Evaluate model on no noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0083
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.01851310022175312   0.011143352836370468   0.01231973897665739   0.08054083585739136   0.008469109423458576   0.008281579241156578
Evaluate model on Gaussian noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0185
Evaluate model on Speckle noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0111
Evaluate model on Salt&Pepper noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0123
Evaluate model on Block noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0812
Evaluate model on Border noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0085
Evaluate model on no noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0083
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.018541118130087852   0.01113847829401493   0.012267285957932472   0.08116327971220016   0.008464016020298004   0.008276286534965038
Evaluate model on Gaussian noise:
79/79 [==============================] - 3s 40ms/step - loss: 0.0185
Evaluate model on Speckle noise:
79/79 [==============================] - 3s 40ms/step - loss: 0.0111
Evaluate model on Salt&Pepper noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0123
Evaluate model on Block noise:
79/79 [==============================] - 3s 42ms/step - loss: 0.0810
Evaluate model on Border noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0085
Evaluate model on no noise:
79/79 [==============================] - 4s 46ms/step - loss: 0.0083
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.01850517839193344   0.011128389276564121   0.012306745164096355   0.08100990951061249   0.008453943766653538   0.008265799842774868
Evaluate model on Gaussian noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.0184
Evaluate model on Speckle noise:
40/40 [==============================] - 2s 61ms/step - loss: 0.0111
Evaluate model on Salt&Pepper noise:
40/40 [==============================] - 3s 72ms/step - loss: 0.0122
Evaluate model on Block noise:
40/40 [==============================] - 3s 69ms/step - loss: 0.0803
Evaluate model on Border noise:
40/40 [==============================] - 3s 79ms/step - loss: 0.0084
Evaluate model on no noise:
40/40 [==============================] - 3s 75ms/step - loss: 0.0082
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.018428154289722443   0.011103675700724125   0.012242501601576805   0.08026185631752014   0.008434167131781578   0.008245213888585567
Evaluate model on Gaussian noise:
20/20 [==============================] - 3s 126ms/step - loss: 0.0186
Evaluate model on Speckle noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.0111
Evaluate model on Salt&Pepper noise:
20/20 [==============================] - 2s 122ms/step - loss: 0.0123
Evaluate model on Block noise:
20/20 [==============================] - 2s 124ms/step - loss: 0.0809
Evaluate model on Border noise:
20/20 [==============================] - 2s 124ms/step - loss: 0.0085
Evaluate model on no noise:
20/20 [==============================] - 2s 123ms/step - loss: 0.0083
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.01856602169573307   0.011122663505375385   0.012291574850678444   0.08091747760772705   0.008481034077703953   0.008293399587273598
Evaluate model on Gaussian noise:
10/10 [==============================] - 2s 216ms/step - loss: 0.0185
Evaluate model on Speckle noise:
10/10 [==============================] - 2s 216ms/step - loss: 0.0111
Evaluate model on Salt&Pepper noise:
10/10 [==============================] - 2s 208ms/step - loss: 0.0123
Evaluate model on Block noise:
10/10 [==============================] - 2s 202ms/step - loss: 0.0808
Evaluate model on Border noise:
10/10 [==============================] - 2s 206ms/step - loss: 0.0085
Evaluate model on no noise:
10/10 [==============================] - 2s 219ms/step - loss: 0.0083
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.018544960767030716   0.011143729090690613   0.01231932733207941   0.08080010861158371   0.008482808247208595   0.008294930681586266
```

## Gauss

```
Averge loss over 6 runs with batch sizes:  [32, 64, 128, 256, 512, 1024]  and a data set size of (10000, 28, 28, 1)
Gauss:  0.015265368235607943
Speckle:  0.07947630683581035
Salt and Pepper:  0.06296769281228383
Block:  0.08571630343794823
Border:  0.049197349697351456
No noise:  0.0276318471878767
Joint average:  0.05337581136781308
```

### The runs

```
Evaluate model on Gaussian noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0152
Evaluate model on Speckle noise:
313/313 [==============================] - 5s 16ms/step - loss: 0.0795
Evaluate model on Salt&Pepper noise:
313/313 [==============================] - 5s 17ms/step - loss: 0.0629
Evaluate model on Block noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0856
Evaluate model on Border noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0492
Evaluate model on no noise:
313/313 [==============================] - 5s 16ms/step - loss: 0.0277
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.01521320641040802   0.07950308918952942   0.06291184574365616   0.08558551967144012   0.049221329391002655   0.02765701897442341
Evaluate model on Gaussian noise:
157/157 [==============================] - 4s 28ms/step - loss: 0.0153
Evaluate model on Speckle noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0794
Evaluate model on Salt&Pepper noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0628
Evaluate model on Block noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0858
Evaluate model on Border noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0492
Evaluate model on no noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0276
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.015287654474377632   0.07944397628307343   0.06283619999885559   0.08578042685985565   0.049189988523721695   0.027626866474747658
Evaluate model on Gaussian noise:
79/79 [==============================] - 3s 44ms/step - loss: 0.0152
Evaluate model on Speckle noise:
79/79 [==============================] - 3s 44ms/step - loss: 0.0795
Evaluate model on Salt&Pepper noise:
79/79 [==============================] - 4s 44ms/step - loss: 0.0630
Evaluate model on Block noise:
79/79 [==============================] - 3s 44ms/step - loss: 0.0855
Evaluate model on Border noise:
79/79 [==============================] - 3s 44ms/step - loss: 0.0491
Evaluate model on no noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0276
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.015249687246978283   0.07954300940036774   0.06304284930229187   0.08550684154033661   0.049127865582704544   0.027567123994231224
Evaluate model on Gaussian noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.0152
Evaluate model on Speckle noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.0798
Evaluate model on Salt&Pepper noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0630
Evaluate model on Block noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.0856
Evaluate model on Border noise:
40/40 [==============================] - 3s 70ms/step - loss: 0.0490
Evaluate model on no noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0274
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.015229162760078907   0.07977953553199768   0.06296952068805695   0.0856398493051529   0.049005936831235886   0.027449870482087135
Evaluate model on Gaussian noise:
20/20 [==============================] - 2s 120ms/step - loss: 0.0153
Evaluate model on Speckle noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.0793
Evaluate model on Salt&Pepper noise:
20/20 [==============================] - 2s 115ms/step - loss: 0.0629
Evaluate model on Block noise:
20/20 [==============================] - 2s 122ms/step - loss: 0.0862
Evaluate model on Border noise:
20/20 [==============================] - 2s 115ms/step - loss: 0.0494
Evaluate model on no noise:
20/20 [==============================] - 2s 116ms/step - loss: 0.0278
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.01533920131623745   0.07930482923984528   0.06294737011194229   0.08618543297052383   0.04936406761407852   0.027788657695055008
Evaluate model on Gaussian noise:
10/10 [==============================] - 2s 208ms/step - loss: 0.0153
Evaluate model on Speckle noise:
10/10 [==============================] - 2s 206ms/step - loss: 0.0793
Evaluate model on Salt&Pepper noise:
10/10 [==============================] - 2s 216ms/step - loss: 0.0631
Evaluate model on Block noise:
10/10 [==============================] - 2s 214ms/step - loss: 0.0856
Evaluate model on Border noise:
10/10 [==============================] - 2s 216ms/step - loss: 0.0493
Evaluate model on no noise:
10/10 [==============================] - 2s 219ms/step - loss: 0.0277
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.01527329720556736   0.07928340137004852   0.06309837102890015   0.08559975028038025   0.04927491024136543   0.027701545506715775
```

## Speckle

```
Averge loss over 6 runs with batch sizes:  [32, 64, 128, 256, 512, 1024]  and a data set size of (10000, 28, 28, 1)
Gauss:  0.09412940094868343
Speckle:  0.008848877778897682
Salt and Pepper:  0.0409656489888827
Block:  0.08254017680883408
Border:  0.1182505339384079
No noise:  0.008200282386193672
Joint average:  0.05882248680831658
```

### The runs

```
Evaluate model on Gaussian noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0940
Evaluate model on Speckle noise:
313/313 [==============================] - 4s 13ms/step - loss: 0.0088
Evaluate model on Salt&Pepper noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0410
Evaluate model on Block noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0827
Evaluate model on Border noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.1182
Evaluate model on no noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0082
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09404078871011734   0.008843369781970978   0.04104122519493103   0.08265160769224167   0.11815717816352844   0.008199123665690422
Evaluate model on Gaussian noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0941
Evaluate model on Speckle noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0089
Evaluate model on Salt&Pepper noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0409
Evaluate model on Block noise:
157/157 [==============================] - 4s 27ms/step - loss: 0.0824
Evaluate model on Border noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.1182
Evaluate model on no noise:
157/157 [==============================] - 4s 27ms/step - loss: 0.0082
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09411529451608658   0.00885467603802681   0.04092670604586601   0.08239787071943283   0.11824887245893478   0.008198686875402927
Evaluate model on Gaussian noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0942
Evaluate model on Speckle noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0088
Evaluate model on Salt&Pepper noise:
79/79 [==============================] - 3s 41ms/step - loss: 0.0410
Evaluate model on Block noise:
79/79 [==============================] - 4s 44ms/step - loss: 0.0826
Evaluate model on Border noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.1184
Evaluate model on no noise:
79/79 [==============================] - 4s 47ms/step - loss: 0.0082
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09420199692249298   0.008838494308292866   0.040977612137794495   0.08264771848917007   0.11843062937259674   0.008197825402021408
Evaluate model on Gaussian noise:
40/40 [==============================] - 3s 69ms/step - loss: 0.0945
Evaluate model on Speckle noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.0088
Evaluate model on Salt&Pepper noise:
40/40 [==============================] - 3s 73ms/step - loss: 0.0408
Evaluate model on Block noise:
40/40 [==============================] - 3s 67ms/step - loss: 0.0822
Evaluate model on Border noise:
40/40 [==============================] - 3s 72ms/step - loss: 0.1188
Evaluate model on no noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0082
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09446339309215546   0.008818717673420906   0.04080427065491676   0.08222362399101257   0.11878731101751328   0.008196132257580757
Evaluate model on Gaussian noise:
20/20 [==============================] - 2s 119ms/step - loss: 0.0939
Evaluate model on Speckle noise:
20/20 [==============================] - 2s 120ms/step - loss: 0.0089
Evaluate model on Salt&Pepper noise:
20/20 [==============================] - 2s 119ms/step - loss: 0.0410
Evaluate model on Block noise:
20/20 [==============================] - 2s 120ms/step - loss: 0.0823
Evaluate model on Border noise:
20/20 [==============================] - 2s 117ms/step - loss: 0.1179
Evaluate model on no noise:
20/20 [==============================] - 2s 120ms/step - loss: 0.0082
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09391732513904572   0.008873280137777328   0.040994368493556976   0.08233452588319778   0.11786140501499176   0.008201885037124157
Evaluate model on Gaussian noise:
10/10 [==============================] - 2s 213ms/step - loss: 0.0940
Evaluate model on Speckle noise:
10/10 [==============================] - 2s 213ms/step - loss: 0.0089
Evaluate model on Salt&Pepper noise:
10/10 [==============================] - 2s 217ms/step - loss: 0.0410
Evaluate model on Block noise:
10/10 [==============================] - 2s 219ms/step - loss: 0.0830
Evaluate model on Border noise:
10/10 [==============================] - 2s 219ms/step - loss: 0.1180
Evaluate model on no noise:
10/10 [==============================] - 2s 222ms/step - loss: 0.0082
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09403760731220245   0.00886472873389721   0.041049711406230927   0.08298571407794952   0.11801780760288239   0.008208041079342365
```

## Salt and Pepper

```
Averge loss over 6 runs with batch sizes:  [32, 64, 128, 256, 512, 1024]  and a data set size of (10000, 28, 28, 1)
Gauss:  0.09293300658464432
Speckle:  0.012453239411115646
Salt and Pepper:  0.009716722493370375
Block:  0.07705215613047282
Border:  0.0891947125395139
No noise:  0.008083195270349583
Joint average:  0.04823883873824444
```

### The runs

```
Evaluate model on Gaussian noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0929
Evaluate model on Speckle noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0124
Evaluate model on Salt&Pepper noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0097
Evaluate model on Block noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0772
Evaluate model on Border noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0892
Evaluate model on no noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0081
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09290750324726105   0.012443398125469685   0.009706958197057247   0.07719368487596512   0.08920473605394363   0.008087818510830402
Evaluate model on Gaussian noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0929
Evaluate model on Speckle noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0125
Evaluate model on Salt&Pepper noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0097
Evaluate model on Block noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0770
Evaluate model on Border noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0892
Evaluate model on no noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0081
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09285734593868256   0.012511598877608776   0.009703649207949638   0.07697813957929611   0.08920691162347794   0.008083260618150234
Evaluate model on Gaussian noise:
79/79 [==============================] - 3s 43ms/step - loss: 0.0930
Evaluate model on Speckle noise:
79/79 [==============================] - 3s 43ms/step - loss: 0.0124
Evaluate model on Salt&Pepper noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0097
Evaluate model on Block noise:
79/79 [==============================] - 3s 44ms/step - loss: 0.0768
Evaluate model on Border noise:
79/79 [==============================] - 4s 46ms/step - loss: 0.0892
Evaluate model on no noise:
79/79 [==============================] - 4s 46ms/step - loss: 0.0081
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09303149580955505   0.012419428676366806   0.009705956093966961   0.07678696513175964   0.08921119570732117   0.008074234239757061
Evaluate model on Gaussian noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.0932
Evaluate model on Speckle noise:
40/40 [==============================] - 3s 67ms/step - loss: 0.0125
Evaluate model on Salt&Pepper noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0097
Evaluate model on Block noise:
40/40 [==============================] - 3s 67ms/step - loss: 0.0768
Evaluate model on Border noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0892
Evaluate model on no noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0081
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09321435540914536   0.012468956410884857   0.009719731286168098   0.07678777724504471   0.08921956270933151   0.008056515827775002
Evaluate model on Gaussian noise:
20/20 [==============================] - 2s 119ms/step - loss: 0.0928
Evaluate model on Speckle noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.0124
Evaluate model on Salt&Pepper noise:
20/20 [==============================] - 2s 116ms/step - loss: 0.0097
Evaluate model on Block noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.0773
Evaluate model on Border noise:
20/20 [==============================] - 2s 115ms/step - loss: 0.0891
Evaluate model on no noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.0081
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09275957196950912   0.012435556389391422   0.00971892662346363   0.0772549957036972   0.08914252370595932   0.0080973906442523
Evaluate model on Gaussian noise:
10/10 [==============================] - 2s 211ms/step - loss: 0.0928
Evaluate model on Speckle noise:
10/10 [==============================] - 2s 209ms/step - loss: 0.0124
Evaluate model on Salt&Pepper noise:
10/10 [==============================] - 2s 216ms/step - loss: 0.0097
Evaluate model on Block noise:
10/10 [==============================] - 2s 214ms/step - loss: 0.0773
Evaluate model on Border noise:
10/10 [==============================] - 2s 216ms/step - loss: 0.0892
Evaluate model on no noise:
10/10 [==============================] - 2s 217ms/step - loss: 0.0081
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.09282776713371277   0.012440497986972332   0.009745113551616669   0.07731137424707413   0.08918334543704987   0.008099951781332493
```

## Block

```
Averge loss over 6 runs with batch sizes:  [32, 64, 128, 256, 512, 1024]  and a data set size of (10000, 28, 28, 1)
Gauss:  0.08899603659907977
Speckle:  0.02175221374879281
Salt and Pepper:  0.045755223681529365
Block:  0.08019895354906718
Border:  0.13495725144942602
No noise:  0.006641502336909373
Joint average:  0.06305019689413409
```

### The runs

```
Evaluate model on Gaussian noise:
313/313 [==============================] - 5s 16ms/step - loss: 0.0889
Evaluate model on Speckle noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0218
Evaluate model on Salt&Pepper noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0458
Evaluate model on Block noise:
313/313 [==============================] - 5s 14ms/step - loss: 0.0802
Evaluate model on Border noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.1349
Evaluate model on no noise:
313/313 [==============================] - 5s 14ms/step - loss: 0.0066
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08893711119890213   0.021792806684970856   0.04583246260881424   0.08023307472467422   0.134870246052742   0.006645929999649525
Evaluate model on Gaussian noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0890
Evaluate model on Speckle noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0218
Evaluate model on Salt&Pepper noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0457
Evaluate model on Block noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0800
Evaluate model on Border noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.1350
Evaluate model on no noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0066
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08903917670249939   0.02175542525947094   0.04571663588285446   0.08001314848661423   0.13496080040931702   0.006641989108175039
Evaluate model on Gaussian noise:
79/79 [==============================] - 3s 36ms/step - loss: 0.0892
Evaluate model on Speckle noise:
79/79 [==============================] - 3s 41ms/step - loss: 0.0217
Evaluate model on Salt&Pepper noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0457
Evaluate model on Block noise:
79/79 [==============================] - 3s 43ms/step - loss: 0.0801
Evaluate model on Border noise:
79/79 [==============================] - 4s 44ms/step - loss: 0.1351
Evaluate model on no noise:
79/79 [==============================] - 4s 46ms/step - loss: 0.0066
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08923836797475815   0.021732915192842484   0.045670460909605026   0.08011806756258011   0.13514013588428497   0.0066341832280159
Evaluate model on Gaussian noise:
40/40 [==============================] - 3s 63ms/step - loss: 0.0892
Evaluate model on Speckle noise:
40/40 [==============================] - 3s 63ms/step - loss: 0.0218
Evaluate model on Salt&Pepper noise:
40/40 [==============================] - 3s 70ms/step - loss: 0.0459
Evaluate model on Block noise:
40/40 [==============================] - 3s 65ms/step - loss: 0.0804
Evaluate model on Border noise:
40/40 [==============================] - 3s 69ms/step - loss: 0.1355
Evaluate model on no noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0066
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.0892220139503479   0.021841559559106827   0.04586074501276016   0.08042748272418976   0.13549205660820007   0.006618859712034464
Evaluate model on Gaussian noise:
20/20 [==============================] - 2s 115ms/step - loss: 0.0887
Evaluate model on Speckle noise:
20/20 [==============================] - 2s 116ms/step - loss: 0.0217
Evaluate model on Salt&Pepper noise:
20/20 [==============================] - 2s 116ms/step - loss: 0.0457
Evaluate model on Block noise:
20/20 [==============================] - 2s 116ms/step - loss: 0.0803
Evaluate model on Border noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.1346
Evaluate model on no noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.0067
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08873865753412247   0.021676354110240936   0.04566570371389389   0.08025667816400528   0.13456860184669495   0.006650141440331936
Evaluate model on Gaussian noise:
10/10 [==============================] - 2s 230ms/step - loss: 0.0888
Evaluate model on Speckle noise:
10/10 [==============================] - 2s 206ms/step - loss: 0.0217
Evaluate model on Salt&Pepper noise:
10/10 [==============================] - 2s 210ms/step - loss: 0.0458
Evaluate model on Block noise:
10/10 [==============================] - 2s 210ms/step - loss: 0.0801
Evaluate model on Border noise:
10/10 [==============================] - 2s 208ms/step - loss: 0.1347
Evaluate model on no noise:
10/10 [==============================] - 2s 212ms/step - loss: 0.0067
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08880089223384857   0.0217142216861248   0.0457853339612484   0.08014526963233948   0.13471166789531708   0.006657910533249378
```
## Border

```
Averge loss over 6 runs with batch sizes:  [32, 64, 128, 256, 512, 1024]  and a data set size of (10000, 28, 28, 1)
Gauss:  0.16764418532450995
Speckle:  0.05668008141219616
Salt and Pepper:  0.05978979542851448
Block:  0.1121079462269942
Border:  0.006683202926069498
No noise:  0.05025981552898884
Joint average:  0.07552750447454552
```

### The runs



```
Evaluate model on Gaussian noise:
313/313 [==============================] - 4s 13ms/step - loss: 0.1677
Evaluate model on Speckle noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0567
Evaluate model on Salt&Pepper noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0597
Evaluate model on Block noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.1118
Evaluate model on Border noise:
313/313 [==============================] - 4s 14ms/step - loss: 0.0067
Evaluate model on no noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0503
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.1677446961402893   0.05671977624297142   0.05971979349851608   0.11182927340269089   0.0066866097040474415   0.05028277263045311
Evaluate model on Gaussian noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.1675
Evaluate model on Speckle noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0566
Evaluate model on Salt&Pepper noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0597
Evaluate model on Block noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.1120
Evaluate model on Border noise:
157/157 [==============================] - 4s 24ms/step - loss: 0.0067
Evaluate model on no noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0503
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.16752812266349792   0.05660199746489525   0.05971886217594147   0.11196544766426086   0.006683715619146824   0.05028645321726799
Evaluate model on Gaussian noise:
79/79 [==============================] - 3s 42ms/step - loss: 0.1676
Evaluate model on Speckle noise:
79/79 [==============================] - 3s 41ms/step - loss: 0.0567
Evaluate model on Salt&Pepper noise:
79/79 [==============================] - 3s 41ms/step - loss: 0.0598
Evaluate model on Block noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.1122
Evaluate model on Border noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0067
Evaluate model on no noise:
79/79 [==============================] - 4s 46ms/step - loss: 0.0503
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.1676073670387268   0.05669436231255531   0.05984383076429367   0.11215797811746597   0.006677983794361353   0.05029376596212387
Evaluate model on Gaussian noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.1675
Evaluate model on Speckle noise:
40/40 [==============================] - 3s 66ms/step - loss: 0.0569
Evaluate model on Salt&Pepper noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0601
Evaluate model on Block noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.1125
Evaluate model on Border noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0067
Evaluate model on no noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0503
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.1675293743610382   0.05693691968917847   0.060080237686634064   0.11245898902416229   0.0066667357459664345   0.05030808970332146
Evaluate model on Gaussian noise:
20/20 [==============================] - 2s 119ms/step - loss: 0.1678
Evaluate model on Speckle noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.0565
Evaluate model on Salt&Pepper noise:
20/20 [==============================] - 2s 117ms/step - loss: 0.0597
Evaluate model on Block noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.1123
Evaluate model on Border noise:
20/20 [==============================] - 2s 116ms/step - loss: 0.0067
Evaluate model on no noise:
20/20 [==============================] - 2s 120ms/step - loss: 0.0502
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.16776274144649506   0.05652719736099243   0.05973752588033676   0.11228331178426743   0.006688094697892666   0.050176799297332764
Evaluate model on Gaussian noise:
10/10 [==============================] - 2s 211ms/step - loss: 0.1677
Evaluate model on Speckle noise:
10/10 [==============================] - 2s 212ms/step - loss: 0.0566
Evaluate model on Salt&Pepper noise:
10/10 [==============================] - 2s 213ms/step - loss: 0.0596
Evaluate model on Block noise:
10/10 [==============================] - 2s 215ms/step - loss: 0.1120
Evaluate model on Border noise:
10/10 [==============================] - 2s 224ms/step - loss: 0.0067
Evaluate model on no noise:
10/10 [==============================] - 2s 218ms/step - loss: 0.0502
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.16769281029701233   0.056600235402584076   0.05963852256536484   0.11195267736911774   0.00669607799500227   0.05021101236343384
```

## No Noise

```
Averge loss over 6 runs with batch sizes:  [32, 64, 128, 256, 512, 1024]  and a data set size of (10000, 28, 28, 1)
Gauss:  0.08762909596165021
Speckle:  0.021531872140864532
Salt and Pepper:  0.045536190271377563
Block:  0.08155869320034981
Border:  0.12130752826730411
No noise:  0.006656824300686519
Joint average:  0.060703367357038795
```

## The runs

```
Evaluate model on Gaussian noise:
313/313 [==============================] - 5s 16ms/step - loss: 0.0875
Evaluate model on Speckle noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0215
Evaluate model on Salt&Pepper noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0455
Evaluate model on Block noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0816
Evaluate model on Border noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.1212
Evaluate model on no noise:
313/313 [==============================] - 5s 15ms/step - loss: 0.0067
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08752431720495224   0.0214751698076725   0.04554625600576401   0.08162914216518402   0.12122131884098053   0.006658631842583418
Evaluate model on Gaussian noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0877
Evaluate model on Speckle noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0216
Evaluate model on Salt&Pepper noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.0455
Evaluate model on Block noise:
157/157 [==============================] - 4s 23ms/step - loss: 0.0814
Evaluate model on Border noise:
157/157 [==============================] - 4s 25ms/step - loss: 0.1213
Evaluate model on no noise:
157/157 [==============================] - 4s 26ms/step - loss: 0.0067
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08765491098165512   0.02157178707420826   0.04547574371099472   0.08139161020517349   0.12130473554134369   0.006655920762568712
Evaluate model on Gaussian noise:
79/79 [==============================] - 3s 43ms/step - loss: 0.0877
Evaluate model on Speckle noise:
79/79 [==============================] - 3s 43ms/step - loss: 0.0216
Evaluate model on Salt&Pepper noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0456
Evaluate model on Block noise:
79/79 [==============================] - 4s 45ms/step - loss: 0.0814
Evaluate model on Border noise:
79/79 [==============================] - 3s 42ms/step - loss: 0.1215
Evaluate model on no noise:
79/79 [==============================] - 4s 46ms/step - loss: 0.0067
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08770231902599335   0.02162802219390869   0.045581378042697906   0.08136208355426788   0.12146987020969391   0.006650547496974468
Evaluate model on Gaussian noise:
40/40 [==============================] - 3s 65ms/step - loss: 0.0879
Evaluate model on Speckle noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.0216
Evaluate model on Salt&Pepper noise:
40/40 [==============================] - 3s 73ms/step - loss: 0.0457
Evaluate model on Block noise:
40/40 [==============================] - 3s 71ms/step - loss: 0.0815
Evaluate model on Border noise:
40/40 [==============================] - 3s 72ms/step - loss: 0.1218
Evaluate model on no noise:
40/40 [==============================] - 3s 68ms/step - loss: 0.0066
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08794842660427094   0.021621815860271454   0.04571319743990898   0.08152057975530624   0.1217939481139183   0.006639999337494373
Evaluate model on Gaussian noise:
20/20 [==============================] - 2s 122ms/step - loss: 0.0875
Evaluate model on Speckle noise:
20/20 [==============================] - 2s 125ms/step - loss: 0.0215
Evaluate model on Salt&Pepper noise:
20/20 [==============================] - 2s 119ms/step - loss: 0.0455
Evaluate model on Block noise:
20/20 [==============================] - 2s 122ms/step - loss: 0.0818
Evaluate model on Border noise:
20/20 [==============================] - 2s 118ms/step - loss: 0.1210
Evaluate model on no noise:
20/20 [==============================] - 2s 121ms/step - loss: 0.0067
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08747584372758865   0.021462365984916687   0.04549913480877876   0.08180592954158783   0.1209600567817688   0.006664975080639124
Evaluate model on Gaussian noise:
10/10 [==============================] - 2s 198ms/step - loss: 0.0875
Evaluate model on Speckle noise:
10/10 [==============================] - 2s 199ms/step - loss: 0.0214
Evaluate model on Salt&Pepper noise:
10/10 [==============================] - 2s 201ms/step - loss: 0.0454
Evaluate model on Block noise:
10/10 [==============================] - 2s 206ms/step - loss: 0.0816
Evaluate model on Border noise:
10/10 [==============================] - 2s 219ms/step - loss: 0.1211
Evaluate model on no noise:
10/10 [==============================] - 2s 218ms/step - loss: 0.0067
Loss: Gauss, speckle, S_P, Block, Border, None
Loss:  0.08746875822544098   0.021432071924209595   0.045401431620121   0.08164281398057938   0.12109524011611938   0.0066708712838590145
```
