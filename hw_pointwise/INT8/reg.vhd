library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;


entity reg is
	generic (
		dataWidth: integer := dataWidth
	);
	port (
		clk: 	in 	std_logic;
		reset:	in 	std_logic;
		enable: in 	std_logic;
		input: 	in 	std_logic_vector(dataWidth-1 downto 0);
		output: out std_logic_vector(dataWidth-1 downto 0)	
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