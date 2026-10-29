// xqmac8 rd, rs1, rs2
//   rd += sum(i=0..7) sext8(rs1[i]) * sext8(rs2[i])
// custom-0 (opcode 0x0b), funct3=1, funct7=0
require_extension('I');

reg_t a = RS1;
reg_t b = RS2;
sreg_t acc = (sreg_t)READ_REG(insn.rd());

for (int i = 0; i < 8; i++) {
  int32_t x = (int8_t)((a >> (8 * i)) & 0xff);
  int32_t y = (int8_t)((b >> (8 * i)) & 0xff);
  acc += (sreg_t)(x * y);
}

WRITE_RD(sext_xlen(acc));