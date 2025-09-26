library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;

entity pwc_adder is
  port (
    accIn  : in  array_256_slv;
    accOut : out float16
  );
end entity;

architecture rtl of pwc_adder is

  component adder
    port (
		a: in float16 ;
		b: in float16 ;
		x: out float16 
    );
  end component;

  signal lv0 : array_128_slv;
  signal lv1 : array_64_slv;
  signal lv2 : array_32_slv;
  signal lv3 : array_16_slv;
  signal lv4 : array_8_slv;
  signal lv5 : array_4_slv;
  signal lv6 : array_2_slv;

begin

  adders_lv0: for i in 1 to 128 generate
    a_lv0: adder port map(a => accIn(2*i - 1), b =>  accIn(2*i), x => lv0(i));
  end generate;

  adders_lv1: for i in 1 to 64 generate
    a_lv1: adder port map(a => lv0(2*i - 1), b =>  lv0(2*i), x => lv1(i));
  end generate;

  adders_lv2: for i in 1 to 32 generate
    a_lv2: adder port map(a => lv1(2*i - 1), b =>  lv1(2*i), x => lv2(i));
  end generate;

  adders_lv3: for i in 1 to 16 generate
    a_lv3: adder port map(a => lv2(2*i - 1), b =>  lv2(2*i), x => lv3(i));
  end generate;

  adders_lv4: for i in 1 to 8 generate
    a_lv4: adder port map(a => lv3(2*i - 1), b =>  lv3(2*i), x => lv4(i));
  end generate;

  adders_lv5: for i in 1 to 4 generate
    a_lv5: adder port map(a => lv4(2*i - 1), b =>  lv4(2*i), x => lv5(i));
  end generate;

  adders_lv6: for i in 1 to 2 generate
    a_lv6: adder port map(a => lv5(2*i - 1), b =>  lv5(2*i), x => lv6(i));
  end generate;

  a_lv7: adder port map(a => lv6(1), b =>  lv6(2), x => accOut);

end architecture;


