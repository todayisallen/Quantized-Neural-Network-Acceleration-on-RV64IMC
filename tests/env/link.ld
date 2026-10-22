/* bare-metal RV64 for Spike
 * 重點：-nostartfiles 不連 crt0，所以 sp 必須自己設，
 *       這裡預留 1 MB stack 並匯出 _stack_top。
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text   : { *(.text.init) *(.text*) }
  .rodata : { *(.rodata*) }
  .data   : { *(.data*) }
  .tohost ALIGN(0x1000) : { *(.tohost) }
  .bss    : { *(.bss*) *(COMMON) }
  . = ALIGN(16);
  _end = .;
  . += 0x100000;
  _stack_top = .;
}
