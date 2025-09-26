library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;

entity reg is
	port (
		clk: 	in 	std_logic;
		reset:	in 	std_logic;
		enable: in 	std_logic;
		input: 	in 	float16;
		output: out float16	
	);
end entity reg;

architecture arcReg of reg is

begin

	process(clk, reset)
	begin
		if (reset = '1') then
			output <= (others=>'0');
		elsif (enable = '1') then
			if (clk'EVENT and clk = '1') then
				output <= input;
			end if;
		end if;
	end process;
	
end architecture arcReg;