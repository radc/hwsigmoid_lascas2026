library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;


entity pwc_mult is
	generic (
		dataWidth: integer := dataWidth
	);
  port (
    img : in array_256_slv;
    k   : in array_256_slv;
    mk  : out array_256_slv
  );
end entity;

architecture rtl of pwc_mult is
  component multiplier
    port (
      a : in std_logic_vector(dataWidth-1 downto 0);
      b : in std_logic_vector(dataWidth-1 downto 0);
      x : out std_logic_vector(dataWidth-1 downto 0)
    );
  end component;
begin
  mults: for i in 1 to 256 generate
    m: multiplier generic map(dataWidth) port map(a => img(i), b => k(i), x => mk(i));
  end generate;

end architecture;


