library ieee;
use ieee.std_logic_1164.all;
use work.float_pkg.all;
--use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;
use work.LUT.all;

entity wsilu_lut_256 is
    generic (
  	LIMIT_1_W : float16 := LIMIT_1;
  	LIMIT_2_W : float16 := LIMIT_2;
  	LIMIT_3_W : float16 := LIMIT_3;
  	LIMIT_4_W : float16 := LIMIT_4;
  	N_1_W: integer := N_1;
  	N_2_W: integer := N_2;
  	N_3_W: integer := N_3;
	N_4_W: integer := N_4
    );
    port (
        clk    : in  std_logic;
        reset    : in  std_logic;
        enable : in  std_logic;
        xIn      : in  float16;
        yOut      : out float16
    );
end entity;
architecture rtl of wsilu_LUT_256  is

constant n256: float16 := to_float(256, 5, 10);
constant n128: float16 := to_float(128, 5, 10);
constant n64: float16 := to_float(64, 5, 10);
constant n192 :integer :=  192;
constant n224 : integer := 224;
signal scaled1   : integer := 0;
signal scaled2   : integer := 0;
signal scaled3   : integer := 0;
signal scaled4   : integer := 0;
signal scaled5   : integer := 0;
signal scaled_val: integer := 0;
signal interval: std_logic_vector(2 downto 0) := (others => '0');
signal x, symmetry : float16 := to_float(0.0, 5, 10);
signal index     : integer range 0 to LUT_SIZE-1 := 0;
signal y, y1     : float16 := to_float(0.0, 5, 10);
component intervalSelector
	port(
		x        : in  float16;
		interval: out std_logic_vector(2 downto 0)
	);
end component;
component mux
	port(
		scaled1 :in integer;
		scaled2 :in integer;
		scaled3 :in integer;
		scaled4 :in integer;
		scaled5 : in integer;
		interval: in std_logic_vector(2 downto 0);
		cout : out integer
		);
	end component;

component multiplier
	port (
		a			:	in 	float16;
		b 			:	in 	float16;
		x			:	out	float16
	);
end component;
begin
	intervalSelec: intervalSelector port map(x, interval);
  	scaled1 <= to_integer(abs(x)  * n256);
 	scaled2 <= N_1_W + to_integer((abs(x) - LIMIT_1_W) * n128);
    	scaled3 <= n192 + to_integer((abs(x) - LIMIT_2_W) * n64);
    	scaled4 <= n224 + to_integer((abs(x) - LIMIT_3_W) * n64);
  	scaled5 <= (LUT_SIZE - 1);

	mux1Module:  mux port map( scaled1, scaled2, scaled3, scaled4,scaled5, interval, scaled_val);

	index <= 0 when (scaled_val <= 0) else
         (LUT_SIZE-1) when (scaled_val > LUT_SIZE-1) else
         scaled_val;
	symmetry <= to_float(1.0, 5, 10) - lut_values(index) when (index > 0 and index < LUT_SIZE) else to_float(0.0, 5, 10);
	
	y1 <= symmetry when x < to_float(0.0, 5, 10) else lut_values(index);
	multiplicador1Module: multiplier port map(x, y1, y);

	process (reset, clk)
	begin
		if (reset = '1') then
			x <= (others => '0');
		elsif (clk'event and clk = '1') then
			x <= xIn;
		end if;
	end process;

	process (reset, clk)
	begin
		if (reset = '1') then
			yOut <= (others => '0');
		elsif (clk'event and clk = '1') then
			yOut <= y;
		end if;
	end process;
end architecture;


