library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;

entity multiplier is
	generic (
		dataWidth: integer := dataWidth
	);
	port (
		a: in std_logic_vector(dataWidth-1 downto 0);
		b: in std_logic_vector(dataWidth-1 downto 0);
		x: out std_logic_vector(dataWidth-1 downto 0)
	);
end multiplier;

architecture multiplier_arc of multiplier is

begin
	x <= std_logic_vector(resize(signed(a)*signed(b), dataWidth));
end multiplier_arc;