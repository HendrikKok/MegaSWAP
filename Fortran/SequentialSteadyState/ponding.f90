module ponding
use globals
implicit none

real(kind=hp), parameter :: soil_parameter = 0.054

    type :: t_soil
        real(kind=hp) :: beta = soil_parameter
        real(kind=hp) :: rch = 0.d0
        real(kind=hp) :: evpt = 0.d0
        real(kind=hp) :: evac = 0.d0
        real(kind=hp) :: cuevpt = 0.d0
        real(kind=hp) :: cuevpt0 = 0.d0
        real(kind=hp) :: cuevac = 0.d0
        real(kind=hp) :: cuevac0 = 0.d0
        real(kind=hp), pointer :: dt => null()
    contains
        procedure, pass :: reset          => t_soil_reset
        procedure, pass :: getActualEvap  => t_soil_getActualEvaporation
        procedure, pass :: update         => t_soil_update
    end type t_soil

    type :: t_ponding
        real(kind=hp) :: zmax
        real(kind=hp) :: area
        real(kind=hp) :: volume = 0.d0
        real(kind=hp) :: stage = 0.d0 
        real(kind=hp) :: soil_resistance
        real(kind=hp) :: max_infiltration_rate
        real(kind=hp), pointer :: dt => null()
        real(kind=hp) :: fponding = 1.d0
    contains
        procedure, pass :: getPondingEvap   => t_ponding_getPondingEvap
        procedure, pass :: getInfiltration  => t_ponding_getInfiltration
        procedure, pass :: getRunoff        => t_ponding_getRunoff
        procedure, pass :: addVolume        => t_ponding_addVolume
        procedure, pass :: addPrecip        => t_ponding_addPrecip
    end type t_ponding

contains
    
    subroutine t_soil_reset(soil)
        class(t_soil), intent(inout) :: soil
        soil%cuevac = 0.d0
        soil%cuevac0 = 0.d0
        soil%cuevpt = 0.d0
        soil%evac = 0.d0
    end subroutine t_soil_reset
    

    subroutine t_soil_update(soil, recharge, pot_evap, dt)
        class(t_soil), intent(inout) :: soil
        real(kind=hp), intent(in)    :: recharge 
        real(kind=hp), intent(in)    :: pot_evap
        real(kind=hp), intent(in)    :: dt
        soil%rch = recharge
        soil%evpt = pot_evap
        soil%dt = dt
    
        soil%cuevac0 = soil%cuevac
        if (soil%rch < soil%evpt) then     ! drying
            soil%cuevpt = soil%cuevpt + (soil%evpt - soil%rch) * soil%dt
            soil%cuevac = merge(soil%cuevpt, (soil%cuevpt**0.5) * soil%beta, soil%cuevpt <= soil%beta**2.)
        else                              ! wetting
            ! integrated rainfall deficit reduced by instant rainfall surplus downto zero
            soil%cuevac = max(soil%cuevac - (soil%rch-soil%evpt) * soil%dt,0.d0)
            soil%cuevpt = merge(soil%cuevac, (soil%cuevac**2.0) / soil%beta**2, soil%cuevac < soil%beta**2)
        endif
    end subroutine t_soil_update
    
    function t_soil_getActualEvaporation(soil) result (evap)
        real(kind=hp)                :: evap 
        class(t_soil), intent(inout) :: soil
        if (soil%rch < soil%evpt) then
            soil%evac = soil%rch + (soil%cuevac - soil%cuevac0) / soil%dt
        else
            soil%evac = soil%evpt
        endif
        evap = soil%evac
    end function t_soil_getActualEvaporation

    subroutine t_ponding_addVolume(ponding, volume, realised)
        class(t_ponding),        intent(inout) :: ponding
        real(kind=hp),           intent(in)    :: volume
        real(kind=hp), optional, intent(out)   :: realised
        real(kind=hp) :: vol0
        vol0 = ponding%volume
        ponding%volume = max(0.d0, ponding%volume + volume)
        ponding%stage = ponding%volume/ponding%area
        if (present(realised)) then
            realised = ponding%volume - vol0 ! return realised change in ponding volume
        endif
    end subroutine t_ponding_addVolume

    subroutine t_ponding_addPrecip(ponding, precip)
        class(t_ponding), intent(inout) :: ponding
        real(kind=hp), intent(in) :: precip
        call ponding%addVolume(precip*ponding%area*ponding%dt)
    end subroutine t_ponding_addPrecip

    function t_ponding_getRunoff(ponding) result (runoff)
        real(kind=hp)                   :: runoff
        class(t_ponding), intent(inout) :: ponding
        real(kind=hp)                   :: excess_volume
        excess_volume = max(0.d0, ponding%stage-ponding%zmax)*ponding%area
        call ponding%addVolume(-excess_volume)
        runoff = excess_volume / ponding%area / ponding%dt
    end function t_ponding_getRunoff

    function t_ponding_getInfiltration(ponding, gwl) result (infiltration)
        real(kind=hp)                   :: infiltration
        class(t_ponding), intent(inout) :: ponding
        real(kind=hp),    intent(in)    :: gwl
        infiltration = ponding%volume / ponding%area / ponding%dt
        if (gwl>ponding%zmax) then
            ponding%volume = 0.d0
            ponding%stage = 0.d0
        else
            infiltration = min(infiltration, ponding%max_infiltration_rate)
            call ponding%addVolume(-infiltration*ponding%area*ponding%dt)
        endif
    end function t_ponding_getInfiltration

    function t_ponding_getPondingEvap(ponding, pet) result (evap)
        real(kind=hp)                   :: evap
        class(t_ponding), intent(inout) :: ponding
        real(kind=hp), intent(in)       :: pet
        real(kind=hp)                   :: vol_evap
        if (ponding%stage>0.d0) then
            evap = pet * ponding%fponding * ponding%dt
            call ponding%addVolume(-evap*ponding%area, vol_evap) 
            evap = -vol_evap/ponding%area ! evap holds the realised evap volume  
        else
            evap = 0.d0
        endif
    end function t_ponding_getPondingEvap

end module ponding