pro dump_one, lun, name, cube, na
    compile_opt idl2
    printf, lun, '#' + name
    for ia=0, na-1 do printf, lun, reform(cube[ia, *]), format='(1000(g0,:," "))'
end

pro dump_cubes, savfile, out=out
    ; Dump the survey cubes from a moon_mass_a_test.sav to a plain-text file so
    ; they can be read/compared outside IDL.
    ;   dump_cubes, '/Users/kevinwagner/Desktop/ExoMoonSim/<rundir>/moon_mass_a_test.sav'
    compile_opt idl2
    restore, savfile     ; -> sigcube, perrcube, amperrcube, masses, as, success_cube[, ntrials]
    if n_elements(out) eq 0 then out = '/Users/kevinwagner/idl/idl_cubes_dump.txt'
    if n_elements(ntrials) eq 0 then ntrials = -1
    na = n_elements(as) & nm = n_elements(masses)

    openw, lun, out, /get_lun
    printf, lun, 'ntrials= ', ntrials
    printf, lun, 'na= ', na
    printf, lun, 'nm= ', nm
    printf, lun, '#as'
    printf, lun, as, format='(1000(g0,:," "))'
    printf, lun, '#masses'
    printf, lun, masses, format='(1000(g0,:," "))'
    dump_one, lun, 'sigcube', sigcube, na
    dump_one, lun, 'perrcube', perrcube, na
    dump_one, lun, 'amperrcube', amperrcube, na
    dump_one, lun, 'detfrac', success_cube, na
    free_lun, lun
    print, 'Wrote cubes to ', out
end
