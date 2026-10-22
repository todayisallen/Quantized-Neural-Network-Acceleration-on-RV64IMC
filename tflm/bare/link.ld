/* bare-metal RV64 的記憶體佈局。
 *
 * 以 tests/env/link.ld（已驗證可用）為基礎，多了四件 TFLM 需要的東西：
 *   1. .init_array —— TFLM 是 C++，有靜態物件要在推論前跑建構子
 *   2. .srodata / .sdata / .sbss —— RISC-V 的 small data 節，
 *      漏了會變成 orphan section，連結器會把它丟到不可預期的位置
 *   3. _bss_start / _bss_end —— cstart() 要自己清 .bss
 *   4. heap —— newlib 的 libm 幾乎不用，但 _sbrk 要有東西可以給
 *
 * 記憶體預算（EfficientNet-Lite0）：
 *   .rodata  約 5.3 MB  模型
 *   .bss     約 1.65 MB tensor arena
 *   heap     256 KB
 *   stack    1 MB
 *   合計     約 8.3 MB，spike -m64（64 MB）夠用
 */

OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
  /* 載入位址由 Makefile 的 -Wl,--defsym=LOAD_BASE=... 指定，只有一個來源。
     0x80000000 跟 tests/env 一致，也是 Spike 的預設 DRAM_BASE。
     之所以能用這個位址，是因為我們不連 newlib —— 它的 libm 是 medlow 建的，
     絕對定址只到 0x7fffffff。詳見 bare/libc_min.cc 開頭。 */
  . = DEFINED(LOAD_BASE) ? LOAD_BASE : 0x80000000;

  /* *(.text.init) 必須寫在 *(.text*) 前面。
     連結器按順序比對 pattern，而 .text.init 也符合 .text*，
     寫反的話 _start 就不會落在 0x80000000。 */
  .text : {
    *(.text.init)
    *(.text .text.*)
  }

  .rodata : ALIGN(16) {
    *(.rodata .rodata.*)
    *(.srodata .srodata.*)
    *(.data.rel.ro .data.rel.ro.*)
  }

  /* 全域建構子。少了這段，__init_array_start/__init_array_end
     會是未定義符號，而且 TFLM 的靜態物件不會初始化。 */
  .init_array : ALIGN(8) {
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN(__preinit_array_end = .);
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN(__init_array_end = .);
  }

  .data : ALIGN(16) {
    /* gp 相對定址的基準。_start 會 la gp, __global_pointer$。
       我們同時用 -mno-relax，所以兩邊都安全。 */
    PROVIDE(__global_pointer$ = . + 0x800);
    *(.data .data.*)
    *(.sdata .sdata.*)
    *(.got .got.plt)
  }

  /* Spike 從 ELF 符號表找 tohost / fromhost（fesvr/htif.cc 會查符號名）。
     單獨一節並 4K 對齊，跟 tests/env 裡驗證過的做法一致。 */
  .tohost ALIGN(0x1000) : { *(.tohost) }

  .bss : ALIGN(16) {
    _bss_start = .;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);        /* cstart() 用 64-bit 寫入清零，結尾必須 8 對齊 */
    _bss_end = .;
  }

  . = ALIGN(16);
  _end = .;              /* _sbrk 的起點 */
  . += 0x40000;          /* 256 KB heap */
  _heap_end = .;
  . += 0x100000;         /* 1 MB stack */
  _stack_top = .;

  /DISCARD/ : {
    *(.eh_frame .eh_frame_hdr)
    *(.comment)
    *(.note .note.*)
    *(.fini_array .fini_array.*)   /* 永遠不會結束，不需要解構子 */
  }
}
