library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_signed.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.float_pkg.all;
--use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;
use work.LUT.all;

entity multiplier is
	port (
		a			:	in 	float16;
		b 			:	in 	float16;
		x			:	out	float16
	);
end entity;

architecture arq of multiplier is

begin
	x <= a*b;
end architecture;
