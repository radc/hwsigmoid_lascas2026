library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;

entity adder is
	port (
		a: in float16 ;
		b: in float16 ;
		x: out float16 
	);
end adder;

architecture adder_arc of adder is

begin
	x <= a+b;
end adder_arc;