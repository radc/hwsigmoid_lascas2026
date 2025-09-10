library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_signed.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.float_pkg.all;

entity multiplicador is
	port (
		a			:	in 	float32;
		b 			:	in 	float32;
		x			:	out	float32
	);
end multiplicador;

architecture multiplicador_arc of multiplicador is

begin

	x <= a*b;

end multiplicador_arc;