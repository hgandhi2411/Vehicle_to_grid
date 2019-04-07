import math
import scipy.integrate
import numpy as np

class Battery:
    def __init__(self, capacity, eff):
        self.start_capacity = capacity
        self.eff = eff
        self.b1 = 0
        self.c2 = 0
        self.soc = 1.0
        self.dod = 1.0
        self.hour_age = 1
        self.cycles = 0
    @property
    def capacity_fade(self):
        Qli = 1.04 - self.b1
        Qsites = 1 - self.c2
        return min(Qli, Qsites)

    @property
    def capacity(self):
        return self.start_capacity * (self.capacity_fade)

    def voc(self, soc):
        # Voc calc reference: Energies 2016, 9, 900
        # Parameters for LMNCO cathodes at 45 degreeC - couldn't find for NCA batteries
        if isinstance(soc, (list, np.ndarray)):
            if soc.any() > 1.0 :
                index = soc > np.array(np.ones(len(soc)))
                soc[not index] = 1.0
        elif soc > 1.0:
            soc = 1.0
        a = 3.535
        b = -0.0571
        c = -0.2847
        d = 0.9475
        m = 1.4
        n = 2
        Voc = a + b * (-np.log(soc))**m + c * soc + d * np.exp(n * (soc - 1))
        return Voc
    def charge(self, rate, T, temperature=298.15, stop_soc = 1.0, subinterval_limit = 10):
        ''' Charge battery at rate for time T in hours, optionally stopping once soc reaches stop point.
        '''
        if self.soc >= stop_soc:
            return 0
        #just to crash on being given a date or something
        T = float(T)
        assert T < 10**5 # make sure not UTC stamp
        assert stop_soc <= 1.0
        #reference quantities
        T_ref = 298.15 	# K
        V_ref = 3.6		#V
        Dod_ref = 1.0
        F = 96485 		# C/mol	Faraday constant
        Rug = 8.314	# J/K/mol

        # fitted parameters for NCA chemistries
        b0 = 1.04
        c0= 1
        b1_ref = 1.794 * 10**(-4)	# 1/day^(0.5)
        c2_ref = 1.074 * 10**(-4)	# 1/day
        Eab1 = 35000				# J/mol
        alpha_b1 = 0.051
        beta_b1 = 0.99
        Eac2 = 35000				# J/mol
        alpha_c2 =  0.023
        beta_c2 = 2.61
        #compute if we will stop due to SOC or time, gives hours of charging
        Teff = min(T, (stop_soc - self.soc) * self.capacity / rate / self.eff)
        #compute N - fractional cycle
        N = Teff * rate * self.eff / self.capacity

        time_grid = np.linspace(self.hour_age, self.hour_age + Teff, subinterval_limit)
        # get soc at each hour of the charging cycle
        soc = self.soc + (time_grid - self.hour_age) * rate * self.eff / self.capacity
        soc[-1] = min(soc[-1], 1 - 10**-5)
        soc[0] = max(10**-5, soc[0])
        voc = self.voc(soc)
        y = 0.5 * time_grid**(-1/2) * np.exp(-Eab1/Rug * (1/temperature - 1/T_ref)) \
            * np.exp(alpha_b1 * F / Rug *(voc/temperature - V_ref/T_ref)) \
            * ((1 + (1 - self.soc)/Dod_ref)**beta_b1)
        #trapezoidal
        self.b1 += b1_ref * np.sum((y[1:] + y[:-1]) * (time_grid[1:] - time_grid[:-1]) * 0.5)
        y =  np.exp(-Eac2/Rug * (1/temperature - 1/T_ref)) \
		* np.exp(alpha_c2 * F / Rug *(voc/temperature - V_ref/T_ref)) \
		* (N * ((1 - self.soc)/Dod_ref)**beta_c2)
        self.c2 += c2_ref * np.sum((y[1:] + y[:-1]) * (time_grid[1:] - time_grid[:-1]) * 0.5)

        #voc = lambda t: self.voc(self.soc + t * rate * self.eff / c)
        # integrate over open cricuit voltage
        # leading term 0.5 t^{-1/2} is to ensure integral results in b1 t^{1/2}
        # / 24 to go to hours?????

        #self.b1 += b1_ref   * scipy.integrate.quad(lambda t: 0.5 * t**(-1/2) * math.e**(-Eab1/Rug * (1/temperature - 1/T_ref)) \
        #    * math.e**(alpha_b1 * F / Rug *(voc(t - self.hour_age)/temperature - V_ref/T_ref)) \
        #    * ((1 + (1 - self.soc)/Dod_ref)**beta_b1), self.hour_age,self.hour_age + #Teff, limit=subinterval_limit)[0]
        # / 24 to go to hours?????
        #self.c2 += c2_ref * scipy.integrate.quad(lambda t: math.e**(-Eac2/Rug * (1/temperature - 1/T_ref)) \
		#* math.e**(alpha_c2 * F / Rug *(voc(t)/temperature - V_ref/T_ref)) \
		#* (N* ((1 - self.soc)/Dod_ref)**beta_c2), 0, Teff, limit=subinterval_limit)[0]

        self.soc += N
        self.hour_age += T
        self.cycles += N

        assert self.b1 < 1 and self.c2 < 1

        return rate * Teff

    def age(self, T, temperature=298.15):
         #just to crash on being given a date or something
        T = float(T)
        assert T < 10**5 # make sure not UTC stamp
        #reference quantities
        T_ref = 298.15 	# K
        V_ref = 3.6		#V
        Dod_ref = 1.0
        F = 96485 		# C/mol	Faraday constant
        Rug = 8.314	# J/K/mol

        # fitted parameters for NCA chemistries
        b0 = 1.04
        c0= 1
        b1_ref = 1.794 * 10**(-4)	# 1/day^(0.5)
        c2_ref = 1.074 * 10**(-4)	# 1/day
        Eab1 = 35000				# J/mol
        alpha_b1 = 0.051
        beta_b1 = 0.99
        Eac2 = 35000				# J/mol
        alpha_c2 =  0.023
        beta_c2 = 2.61

        voc = self.voc(self.soc)
        # / 24 to go to hours?????
        self.b1 += b1_ref  * \
            ((T + self.hour_age)**(1/2) - self.hour_age**(1/2)) * \
             math.e**(-Eab1/Rug * (1/temperature - 1/T_ref)) \
            * math.e**(alpha_b1 * F / Rug *(voc/temperature - V_ref/T_ref)) \
            * ((1 + (1 - self.soc)/Dod_ref)**beta_b1)

        self.hour_age += T

        assert self.b1 < 1


    def discharge(self, rate, T, stop_soc = 0.1, temperature=298.15, eff=None):
        '''
        Discharge to T or stop_soc.
        '''
        if eff is None:
            eff = self.eff
        Teff = min(T, (self.soc - stop_soc) * self.capacity / rate /eff)
        self.soc -= Teff * rate *eff / self.capacity
        #should probably integrate over SOC here!
        self.age(T)
        return Teff * rate
