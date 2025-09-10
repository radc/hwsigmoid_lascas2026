library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;

entity mux is 
    generic(
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
        c_s15 : real
    );
    port(
        intervalo: in std_logic_vector(4 downto 0);
        c : out float32
    );
end mux;

architecture mux_architecture of mux is

    constant c_s1 : float32 := to_float(c_s1, 8, 23);
    constant c_s2 : float32 := to_float(c_s2, 8, 23);
    constant c_s3 : float32 := to_float(c_s3, 8, 23);
    constant c_s4 : float32 := to_float(c_s4, 8, 23);
    constant c_s5 : float32 := to_float(c_s5, 8, 23);
    constant c_s6 : float32 := to_float(c_s6, 8, 23);
    constant c_s7 : float32 := to_float(c_s7, 8, 23);
    constant c_s8 : float32 := to_float(c_s8, 8, 23);
    constant c_s9 : float32 := to_float(c_s9, 8, 23);
    constant c_s10 : float32 := to_float(c_s10, 8, 23);
    constant c_s11 : float32 := to_float(c_s11, 8, 23);
    constant c_s12 : float32 := to_float(c_s12, 8, 23);
    constant c_s13 : float32 := to_float(c_s13, 8, 23);
    constant c_s14 : float32 := to_float(c_s14, 8, 23);
    constant c_s15 : float32 := to_float(c_s15, 8, 23);
    constant c_s16 : float32 := to_float(c_s16, 8, 23);
    
begin
    
    with intervalo select 
        c <=    c_s1    when "00001",
                c_s2    when "00000",
                c_s3    when "00011",
                c_s4    when "00010",
                c_s5    when "00101",
                c_s6    when "00100",
                c_s7    when "00111",
                c_s8    when "00110",
                c_s9    when "01001",
                c_s10   when "01000",
                c_s11   when "01011",
                c_s12   when "01010",
                c_s13   when "01101",
                c_s14   when "01100",
                c_s15   when "01111",
                c_s16   when "01110",
    
end architecture mux_architecture;