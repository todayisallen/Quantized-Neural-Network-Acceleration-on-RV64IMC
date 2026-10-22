# 移植正確性驗證（多輸入）

模型 `model.tflite`，arena 1713 KiB，產生時間 2026-10-19 13:25

輸入是固定種子的隨機 uint8（`[1,224,224,3]`，150,528 bytes）。
刻意不用真實照片：這一步驗的是整數運算的等價性，而隨機資料會讓每層的
激活值鋪滿動態範圍，比平滑的照片更容易踩到 requantize 的平手點。

| seed | vs LiteRT 不一致 | 最大差 | top-5 同序 | 同集合 | 判定 | x86 自比 | RV64 vs x86 | 耗時 |
|---:|---:|---:|:--:|:--:|:--|:--|:--|---:|
| 0 | 11 | 2 | 5/5 | 5/5 | CLOSE | BITEXACT | BITEXACT | 113s |
| 1 | 12 | 4 | 5/5 | 5/5 | DIVERGENT | BITEXACT | BITEXACT | 111s |
| 2 | 6 | 3 | 5/5 | 5/5 | DIVERGENT | BITEXACT | BITEXACT | 113s |
| 3 | 8 | 4 | 3/5 | 5/5 | DIVERGENT | BITEXACT | BITEXACT | 111s |

## 怎麼讀

- **vs LiteRT**：兩套獨立實作的差異，來自 requantize 平手值的進位策略不同，
  隨網路深度累積。差異集中在 ±1~2 LSB 屬正常；top-5 不變表示不影響分類。
- **x86 自比**：把 TFLM-x86 的輸出存下來當 golden 再跟自己比，必須 BITEXACT。
  這一欄只是確認換 golden 的工具沒寫錯，不是獨立的結果。
- **RV64 vs x86**：同一套 C++ 在兩個架構上的結果，**必須 BITEXACT**。
  這是移植正確性的驗收條件，零容忍。
