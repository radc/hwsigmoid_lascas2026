library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;


entity pwc_mult is
  port (
    img : in array_256_slv;
    k   : in array_256_slv;
    mk  : out array_256_slv
  );
end entity;

architecture rtl of pwc_mult is
  component multiplier
    port (
      a : in float16 ;
      b : in float16 ;
      x : out float16 
    );
  end component;
begin
  mults: for i in 1 to 256 generate
    m: multiplier port map(a => img(i), b => k(i), x => mk(i));
  end generate;

end architecture;


