library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;

entity multiplier is
	port (
		a: in float16 ;
		b: in float16 ;
		x: out float16 
	);
end multiplier;

architecture multiplier_arc of multiplier is

begin
	x <= a*b;
end multiplier_arc;