import bcrypt from "bcryptjs";
import { supabase } from "./supabaseClient";

export async function signUpUser({ firstname, lastname, email, password }) {
  const hashedPassword = await bcrypt.hash(password, 10);

  const { data, error } = await supabase
    .from("user")
    .insert([{ firstname, lastname, email, password: hashedPassword }]);

  if (error) return { success: false, error };
  return { success: true, data };
}

export async function signInUser({ email, password }) {
  const { data, error } = await supabase
    .from("user")
    .select("*")
    .eq("email", email)
    .single();

  if (error || !data)
    return { success: false, error: "Invalid email or password" };

  const passwordMatch = await bcrypt.compare(password, data.password);

  if (!passwordMatch)
    return { success: false, error: "Invalid email or password" };

  return { success: true, data };
}
