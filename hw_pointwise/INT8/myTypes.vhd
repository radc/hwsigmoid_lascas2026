library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package myTypes is
	constant dataWidth: integer := 8;

	type array_256_slv is array (1 to 256) of std_logic_vector(dataWidth-1 downto 0);
	type array_128_slv is array (1 to 128) of std_logic_vector(dataWidth-1 downto 0);
	type array_64_slv is array (1 to 64) of std_logic_vector(dataWidth-1 downto 0);
	type array_32_slv is array (1 to 32) of std_logic_vector(dataWidth-1 downto 0);
	type array_16_slv is array (1 to 16) of std_logic_vector(dataWidth-1 downto 0);
	type array_9_slv is array (1 to 9) of std_logic_vector(dataWidth-1 downto 0);
	type array_8_slv is array (1 to 8) of std_logic_vector(dataWidth-1 downto 0);
	type array_4_slv is array (1 to 4) of std_logic_vector(dataWidth-1 downto 0);
	type array_2_slv is array (1 to 2) of std_logic_vector(dataWidth-1 downto 0);


end myTypes;
