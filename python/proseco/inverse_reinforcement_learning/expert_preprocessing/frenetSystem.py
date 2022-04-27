import numpy as np


class FrenetSystem(object):
    def __init__(self):
        self.curve = []
        self.curve_vector_list = []
        self.curve_distance_list = []

    def add_reference_curve(self, curve):
        """Adds a curve in cartesian coordinate system [(x1,y1),(x2,y2)] and caluculate necessary curve variables

        Arguments:
            curve [tuple(x,y)] -- list of tuples of x,y coordinates specifying a curve in the plane
        """
        self.curve = curve
        self.calculate_curve_lists()

    def calculate_curve_lists(self):
        """Calcululates for self.curve (curve in cartesian coordinate system [(x1,y1),(x2,y2)]) the list of vectors of two sucessing points (xi,yi),(xi+1,yi+1)
        and a list of the distances of these vectors.
        """
        number_of_points = len(self.curve)
        self.curve_vector_list = []
        self.curve_distance_list = []
        for i in range(number_of_points - 1):
            vector = self.vector_between_points(self.curve[i + 1], self.curve[i])
            distance = self.distance(self.curve[i + 1], self.curve[i])
            self.curve_vector_list.append(vector)
            self.curve_distance_list.append(distance)

    def distance(self, point1, point2):
        """Calculates the distance of two points (x1,y1) and (x2,y2)

        Arguments:
            point1 tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane
            point2 tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane

        Return:
            float -- distance between the two points
        """
        x1, y1 = point1
        x2, y2 = point2
        distance = np.sqrt(np.power(x1 - x2, 2.0) + np.power(y1 - y2, 2.0))
        return distance

    def vector_between_points(self, forward_point, backward_point):
        """Calculates the vector between two points (x1,y1) and (x2,y2)

        Arguments:
            forward_point tuple(x,y) --  tuple of x,y coordinates specifying the front point of the vector (where the arrow is pointed to)
            backward_point tuple(x,y) --  tuple of x,y coordinates specifying the back point of the vector (where the vector starts)

        Return:
            tuple(x,y) -- tuple specifying the vector between the two points
        """
        x1, y1 = forward_point
        x2, y2 = backward_point
        return x1 - x2, y1 - y2

    def scalar_product(self, point1, point2):
        """Calculates the standard scalar prodcut between two points (x1,y1) and (x2,y2)

        Arguments:
            point1 tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane
            point2 tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane

        Return:
            float -- scalar product between the two points
        """
        x1, y1 = point1
        x2, y2 = point2
        return x1 * x2 + y1 * y2

    def find_nearest_pair_in_curve(self, point):
        """Searches the closest two points (x1,y1) and (x2,y2) from the curve for a given point (x,y)

        Arguments:
            point tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane

        Return:
            forward_point tuple(x,y) -- one of the two closest points of the curve (the one with greater frenet x coordinate)
            backward_point tuple(x,y) -- one of the two closest points of the curve (the one with smaller frenet x coordinate)
            forward_point_index int -- index of the forward_point inside the curve list
            backward_point_index -- index of the backward_point inside the curve list

        """
        nearest_points = []
        index = 0
        for curve_point in self.curve:
            distance = self.distance(curve_point, point)
            nearest_points.append((distance, index, curve_point))
            index += 1

        def first_element(nearest_point):
            return nearest_point[0]

        nearest_points.sort(key=first_element)
        sorted_nearest_points = nearest_points
        _, point1_index, point1 = sorted_nearest_points[0]
        _, point2_index, point2 = sorted_nearest_points[1]
        if point1_index > point2_index:
            forward_point = point1
            forward_point_index = point1_index
            backward_point = point2
            backward_point_index = point2_index
        else:
            forward_point = point2
            forward_point_index = point2_index
            backward_point = point1
            backward_point_index = point1_index
        return forward_point, backward_point, forward_point_index, backward_point_index

    def point_addition(self, point1, point2):
        """Defines the addition of two points (x1,y1) and (x2,y2)

        Arguments:
            point1 tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane
            point2 tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane

        Return:
            tuple(x,y) -- addition of the two points
        """

        x1, y1 = point1
        x2, y2 = point2
        return x1 + x2, y1 + y2

    def point_subtract(self, point1, point2):
        """Defines the subtraction of two points (x1,y1) and (x2,y2)

        Arguments:
            point1 tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane
            point2 tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane

        Return:
            tuple(x,y) -- subtraction of the two points
        """
        x1, y1 = point1
        x2, y2 = point2
        return x1 - x2, y1 - y2

    def point_scalar_mult(self, scalar, point):
        """Defines the scalar multiplication of a scalar with a point (x,y)

        Arguments:
            point tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane
            scalar float -- scalat

        Return:
            tuple(x,y) -- point
        """
        x, y = point
        return scalar * x, scalar * y

    def project_xy_to_frenet(self, point):
        """Main method of class: Calculates for a given point in cartesian coordinates the frenet coordinates relative to the defined curve

        Arguments:
            point tuple(x,y) --  tuple of x,y coordinates specifying a point in the plane in cartesian coordinates

        Return:
            tuple(x,y) -- tuple of x,y coordinates specifying a point relative to the curve in frenet coordinates
        """
        (
            forward_point,
            backward_point,
            forward_point_index,
            backward_point_index,
        ) = self.find_nearest_pair_in_curve(point)
        x_tilde_backward = 0.0
        if backward_point_index > 0:
            for i in range(backward_point_index):
                x_tilde_backward += self.curve_distance_list[i]
        vector_forward_backward = self.curve_vector_list[backward_point_index]
        assert vector_forward_backward[0] == (forward_point[0] - backward_point[0])

        t = (
            self.scalar_product(point, vector_forward_backward)
            - self.scalar_product(backward_point, vector_forward_backward)
        ) / self.scalar_product(vector_forward_backward, vector_forward_backward)
        if not (t >= 0.0 and t <= 1.0):
            return None, None
        # assert(t>=0.0 and t<=1.0)
        point_perpendicular = self.point_addition(
            backward_point, self.point_scalar_mult(t, vector_forward_backward)
        )
        x_tilde = x_tilde_backward + self.distance(point_perpendicular, backward_point)
        perpendiclar_vector = self.point_subtract(point, point_perpendicular)
        y_tilde = np.sqrt(self.scalar_product(perpendiclar_vector, perpendiclar_vector))
        return x_tilde, y_tilde


if __name__ == "__main__":
    system = FrenetSystem()
    curve = [(1.0, 1.0), (1.5, 1.5), (2.5, 1.5), (3.0, 1.0)]
    point = (2.0, 1.0)
    system.add_reference_curve(curve)
    print(system.project_xy_to_frenet(point))
