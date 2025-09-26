library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;


entity pwconv_unit is --PointWise Convolution para um filtro
	generic (
		dataWidth: integer := dataWidth
	);
  port (
      clk: in std_logic;
      reset:in std_logic;
      enable: in std_logic;
    -- Entrada de 1 pixel da imagem de cada canal (256)
	pIn : in array_256_slv;
    -- kernel 1x1 256 canais
	kIn : in array_256_slv;
    -- Saída da convolução (1 valor)
    	conv_out : out std_logic_vector(dataWidth-1 downto 0)
  );
end entity;

architecture rtl of pwconv_unit is

component pwc_adder is
	generic (
		dataWidth: integer := dataWidth
	);
  port (
    accIn  : in  array_256_slv;
    accOut : out std_logic_vector(dataWidth-1 downto 0)
  );
end component;
component pwc_mult is
	generic (
		dataWidth: integer := dataWidth
	);
  port (
    img : in array_256_slv;
    k   : in array_256_slv;
    mk  : out array_256_slv
  );
end component;

component reg
	generic (
		dataWidth: integer := dataWidth
	);
   port (
      clk: in std_logic;
      reset:in std_logic;
      enable: in std_logic;
      input: in std_logic_vector(dataWidth-1 downto 0);
      output: out std_logic_vector(dataWidth-1 downto 0)
);
end component;

  signal  multVals_pIn : array_256_slv;
  signal inReg, outReg: std_logic_vector(dataWidth-1 downto 0);

begin
    m: pwc_mult generic map(dataWidth) port map(pIn, kIn,  multVals_pIn);
    accStage: pwc_adder generic map(dataWidth) port map( multVals_pIn, inReg);
    regModule: reg generic map(dataWidth) port map (clk, reset, enable, inReg, outReg);
    conv_out <= outReg;

end architecture;


