(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a h f k e l b)
(:init 
(harmony)
(planet a)
(planet h)
(planet f)
(planet k)
(planet e)
(planet l)
(planet b)
(province a)
(province h)
(province f)
(province k)
(province e)
(province l)
(province b)
)
(:goal
(and
(craves a h)
(craves h f)
(craves f k)
(craves k e)
(craves e l)
(craves l b)
)))