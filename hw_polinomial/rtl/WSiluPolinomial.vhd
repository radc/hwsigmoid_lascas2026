library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;

entity WSiluPolinomial is
	port (
		clk 	:	in 	std_logic;
		enable	:	in 	std_logic;
		reset	:	in 	std_logic;
		xIn		: 	in 	float32;
		yOut	: 	out float32
		);
end WSiluPolinomial;

architecture WSiluPolinomial_arc of WSiluPolinomial is

	component multiplicador
		port (
			a			:	in 	float32;
			b 			:	in 	float32;
			x			:	out	float32
		);
	end component;

	component somador
		port (
			a: in float32;
			b: in float32;
			c: in float32;
			x: out float32
		);
	end component;

	signal	x			: float32;
	signal	x_quadrado	: float32;
	signal	y			: float32;

begin

	Multiplicador1Module: multiplicador port map(x, x, x_quadrado);
	-- Multiplicador2Module: multiplicador port map(m, k1, mk1);
	-- Multiplicador3Module: multiplicador port map(m, k1, mk1);
	-- somadorModule: somador port map(m, k1, mk1);

	-- registrador de entrada
	process (reset, clk)
	begin
		if (reset = '1') then
			x <= (others => '0');
		elsif (clk'event and clk = '1') then
			x <= xIn;
		end if;
	end process;

	-- registrador de saída
	process (reset, clk)
	begin
		if (reset = '1') then
			yOut <= (others => '0');
		elsif (clk'event and clk = '1') then
			yOut <= y;
		end if;
	end process;

end WSiluPolinomial_arc;