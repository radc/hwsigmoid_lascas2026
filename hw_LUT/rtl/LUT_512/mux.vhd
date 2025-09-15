library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;
--use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;
use work.LUT.all;

entity mux is 
    port(
  	scaled1  :in integer;
  	scaled2  :in integer;
  	scaled3  :in integer;
  	scaled4  :in integer;
  	scaled5 :in integer;
	interval: in std_logic_vector(2 downto 0);
        cout : out integer
    );
end entity;

architecture arch of mux is

    
begin
    
with interval select 
        cout <=    0    when "000",
                scaled1    when "001",
                scaled2    when "010",
                scaled3    when "011",
                scaled4    when "100",
                scaled5    when "101",
                scaled5     when others;   
end architecture;
