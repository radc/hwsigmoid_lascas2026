library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_signed.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.float_pkg.all;

entity somador is
	port (
		a: in float16;
		b: in float16;
        c: in float16;
		x: out float16
	);
end somador;

architecture somador_arc of somador is

begin

	x <= a+b+c;

end somador_arc;