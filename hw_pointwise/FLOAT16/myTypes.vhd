library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fixed_pkg.all;
use work.float_pkg.all;

package myTypes is
	subtype float16 is UNRESOLVED_float(5 downto -10);
 -- 5 bits de expoente e 10 de mantissa
	type array_256_slv is array (1 to 256) of float16;
	type array_128_slv is array (1 to 128) of float16;
	type array_64_slv is array (1 to 64) of float16;
	type array_32_slv is array (1 to 32) of float16;
	type array_16_slv is array (1 to 16) of float16;
	type array_9_slv is array (1 to 9) of float16;
	type array_8_slv is array (1 to 8) of float16;
	type array_4_slv is array (1 to 4) of float16;
	type array_2_slv is array (1 to 2) of float16;


end myTypes;
