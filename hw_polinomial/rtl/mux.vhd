library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;

entity mux is 
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
end mux;

architecture mux_architecture of mux is

    constant c_s0_float : float16 := to_float(c_s0, 4, 10);
    constant c_s1_float : float16 := to_float(c_s1, 4, 10);
    constant c_s2_float : float16 := to_float(c_s2, 4, 10);
    constant c_s3_float : float16 := to_float(c_s3, 4, 10);
    constant c_s4_float : float16 := to_float(c_s4, 4, 10);
    constant c_s5_float : float16 := to_float(c_s5, 4, 10);
    constant c_s6_float : float16 := to_float(c_s6, 4, 10);
    constant c_s7_float : float16 := to_float(c_s7, 4, 10);
    constant c_s8_float : float16 := to_float(c_s8, 4, 10);
    constant c_s9_float : float16 := to_float(c_s9, 4, 10);
    constant c_s10_float : float16 := to_float(c_s10, 4, 10);
    constant c_s11_float : float16 := to_float(c_s11, 4, 10);
    constant c_s12_float : float16 := to_float(c_s12, 4, 10);
    constant c_s13_float : float16 := to_float(c_s13, 4, 10);
    constant c_s14_float : float16 := to_float(c_s14, 4, 10);
    constant c_s15_float : float16 := to_float(c_s15, 4, 10);
    constant c_s16_float : float16 := to_float(c_s16, 4, 10);
    constant c_sx_float : float16 := to_float(c_sx, 4, 10);
    
begin
    
    with intervalo select 
        c <=    c_s0_float    when "00000",
                c_s1_float    when "00001",
                c_s2_float    when "00010",
                c_s3_float    when "00011",
                c_s4_float    when "00100",
                c_s5_float    when "00101",
                c_s6_float    when "00110",
                c_s7_float    when "00111",
                c_s8_float    when "01000",
                c_s9_float    when "01001",
                c_s10_float   when "01010",
                c_s11_float   when "01011",
                c_s12_float   when "01100",
                c_s13_float   when "01101",
                c_s14_float   when "01110",
                c_s15_float   when "01111",
                c_s16_float   when "10000",
                c_sx_float    when others;
    
end architecture mux_architecture;