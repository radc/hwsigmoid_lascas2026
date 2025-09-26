library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;

entity adder is
	generic (
		dataWidth: integer := dataWidth
	);
	port (
		a: in std_logic_vector(dataWidth-1 downto 0);
		b: in std_logic_vector(dataWidth-1 downto 0);
		x: out std_logic_vector(dataWidth-1 downto 0) 
	);
end adder;

architecture adder_arc of adder is

begin
x <= std_logic_vector(resize(signed(a)+signed(b), dataWidth));
end adder_arc;