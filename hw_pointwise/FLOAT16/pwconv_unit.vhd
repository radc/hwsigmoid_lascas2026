library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;

entity pwconv_unit is --PointWise Convolution para um filtro
  port (
      clk: in std_logic;
      reset:in std_logic;
      enable: in std_logic;
    -- Entrada de 1 pixel da imagem de cada canal (256)
	pIn : in array_256_slv;
    -- kernel 1x1 256 canais
	kIn : in array_256_slv;
    	conv_out : out float16
  );
end entity;

architecture rtl of pwconv_unit is

component pwc_adder is
  port (
    accIn  : in  array_256_slv;
    accOut : out float16
  );
end component;
component pwc_mult is
  port (
    img : in array_256_slv;
    k   : in array_256_slv;
    mk  : out array_256_slv
  );
end component;

component reg
   port (
      clk: in std_logic;
      reset:in std_logic;
      enable: in std_logic;
      input: in float16;
      output: out float16
);
end component;

  signal  multVals_pIn : array_256_slv;
  signal inReg, outReg: float16;

begin
    m: pwc_mult port map(pIn, kIn,  multVals_pIn);
    accStage: pwc_adder  port map( multVals_pIn, inReg);
    regModule: reg port map (clk, reset, enable, inReg, outReg);
    conv_out <= outReg;

end architecture;


