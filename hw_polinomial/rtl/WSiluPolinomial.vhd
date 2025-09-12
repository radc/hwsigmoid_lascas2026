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
		xIn		: 	in 	float16;
		yOut	: 	out float16
		);
end WSiluPolinomial;

architecture WSiluPolinomial_arc of WSiluPolinomial is

	component getIntervalo
		port(
			x        : in  float16;
			intervalo: out std_logic_vector(4 downto 0)
		);
	end component;

	component mux
		generic(
			c_s0 : real;
			c_s1 : real;
			c_s2 : real;
			c_s3 : real;
			c_s4 : real;
			c_s5 : real;
			c_s6 : real;
			c_s7 : real;
			c_s8 : real;
			c_s9 : real;
			c_s10 : real;
			c_s11 : real;
			c_s12 : real;
			c_s13 : real;
			c_s14 : real;
			c_s15 : real;
			c_s16 : real;
			c_sx  : real
		);
		port(
			intervalo: in std_logic_vector(4 downto 0);
			c : out float16
		);
	end component;

	component multiplicador
		port (
			a			:	in 	float16;
			b 			:	in 	float16;
			x			:	out	float16
		);
	end component;

	component somador
		port (
			a: in float16;
			b: in float16;
			c: in float16;
			x: out float16
		);
	end component;

	signal	x					: float16;
	signal	intervalo			: std_logic_vector(4 downto 0);
	signal	x_quadrado			: float16;
	signal	c_x_quadrado		: float16;
	signal	c_times_x_quadrado	: float16;
	signal	c_x					: float16;
	signal	c_times_x			: float16;
	signal	c					: float16;
	signal	y					: float16;

begin

	getIntervaloModule: getIntervalo port map(x, intervalo);

	mux1Module: mux generic map(0.0, -0.00947, -0.03964, -0.07245, -0.01180, 0.31836, 0.87061, 0.87061, 0.31787, -0.01367, -0.07178, -0.07483, 0.27051, 0.26294, 0.24866, 0.22717, 0.01075, 0.0) 
				 	port map(intervalo, c_x_quadrado);
	mux2Module: mux generic map(0.0, -0.03897, -0.12683, -0.19702, -0.11218, 0.20410, 0.48315, 0.51709, 0.79639, 1.11426, 1.19531, 1.20508, 0.33130, 0.33179, 0.33203, + 0.33252, + 0.96826, 1.0) 
				 	port map(intervalo, c_x);
	mux3Module: mux generic map(0.0, -0.04077, -0.10498, -0.14258, -0.11292, -0.03668, -0.00039, -0.00039, -0.03674, -0.11359, -0.14172, -0.14819, 0.40454, 0.41650, 0.44238, 0.48633, 0.02046, 0.0) 
				 	port map(intervalo, c);

	multiplicador1Module: multiplicador port map(x, x, x_quadrado);
	multiplicador2Module: multiplicador port map(x_quadrado, c_x_quadrado, c_times_x_quadrado);
	multiplicador3Module: multiplicador port map(x, c_x, c_times_x);

	somadorModule: somador port map(c_times_x_quadrado, c_times_x, c, y);

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