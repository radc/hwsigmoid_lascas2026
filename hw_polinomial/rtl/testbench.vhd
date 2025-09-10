library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;
use work.fixed_pkg.all;
use std.textio.all;

entity testbench is
end testbench;

architecture testbench_arc of testbench is

	signal clk 		:	 std_logic;
	signal enable	:	 std_logic;
	signal reset	:	 std_logic;
	signal x		: 	 float32;
	signal y		: 	 float32;

	component WSiluPolinomial is
		port (
			clk 	:	in 	std_logic;
			enable	:	in 	std_logic;
			reset	:	in 	std_logic;
			xIn		: 	in 	float32;
			yOut	: 	out float32
			);
	end component;

begin

	DUT: WSiluPolinomial port map(clk, enable, reset, x, y);

	clockProcess: process -- process para controlar o clock
	begin
		clk <= '0';
		wait for 5ns;
		clk <= '1';
		wait for 5ns;
	end process clockProcess;

	resetProcess: process -- process para controlar o reset
	begin
--		reset <= '1';
--		wait for 7ns;
		reset <= '0';
		wait;
	end process resetProcess;

	enableProcess: process -- process para controlar o enable
	begin
		enable <= '0';
		wait for 7ns;
		enable <= '1';
		wait;
	end process enableProcess;

	x <= "01000000010000000000000000000000";

	end testbench_arc;